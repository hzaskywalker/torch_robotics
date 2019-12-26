# Baisic GNN forward engine
# The code containts some model similar to ``Graph Networks as Learnable Physics Engines for Inference and Control''

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter_
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.data import Data

from robot.utils.models import fc
from robot.utils.normalizer import Normalizer
from robot.utils.trainer import AgentBase
from robot.utils.quaternion import qmul
from robot.utils import tocpu
from robot.utils import rot6d

import numpy as np


class Concat(nn.Module):
    def __init__(self, main):
        super(Concat, self).__init__()
        self.main = main

    def forward(self, *input: Any):
        input = [i.view(i.shape[0], -1) for i in input]
        return self.main(torch.cat(input, dim=1))


def mlp(inp_dim, oup_dim, feature, num_layer, batch_norm):
    assert num_layer < 10
    if num_layer == 1:
        return fc(inp_dim, oup_dim)
    return nn.Sequential(*([fc(inp_dim, feature, relu=True, batch_norm=batch_norm)] +
                         [fc(feature, feature, relu=True, batch_norm=batch_norm) for i in range(num_layer-2)]+
                         [fc(feature, oup_dim, relu=False, batch_norm=False)]))


def batch_graphs(batch_size, n, graph):
    """
    :return: graph, batch
    """
    batch = torch.arange(batch_size)[:, None].expand(n).view(-1).contiguous()
    graph = graph[None,:] + (torch.arange(batch_size)[:, None] * n)
    graph = graph.permute(1, 0, 2).view(2, -1).contiguous()
    return graph, batch


class GNBlock(nn.Module):
    def __init__(self, node_channels, edge_channels, layers, mid_channels, global_channels=None,
                 output_node_channels=None, output_edge_channels=None, output_global_channels=None):
        super(GNBlock, self).__init__()
        if output_node_channels is None:
            output_node_channels = node_channels
        if output_edge_channels is None:
            output_edge_channels = mid_channels
        if output_global_channels is None:
            output_global_channels = global_channels

        self.edge_mlp = mlp(global_channels + node_channels * 2 + edge_channels,
                            output_edge_channels, mid_channels, layers, batch_norm=False)
        self.node_mlp = mlp(global_channels + node_channels + edge_channels,
                            output_node_channels, mid_channels, layers, batch_norm=False)


        if output_global_channels is not None:
            self.global_mlp = mlp(global_channels + output_node_channels + output_edge_channels,
                                  output_global_channels, mid_channels, layers, batch_norm=False)
        else:
            self.global_mlp = None

    def forward(self, node, edge, edge_index: torch.LongTensor, batch, g=None):
        rol, col = edge_index

        edge_inp = [node[rol], node[col], edge]
        if g is not None:
            edge_batch = batch[rol]
            edge_inp.append(g[edge_batch])
        edge_inp = torch.cat(edge_inp, dim=1)
        edge = self.edge_mlp(edge_inp)


        node_inp = [node, scatter_('add', rol, edge, dim=0)]
        if g is not None:
            node_inp.append(g[batch])
        node_inp = torch.cat(node_inp, dim=1)
        node = self.node_mlp(node_inp)

        if self.global_mlp is not None:
            global_inp = [scatter_('add', batch, node, dim=0), scatter_('edge', batch[rol], edge, dim=0)]
            if g is not None:
                global_inp.append(global_inp)
            g = self.global_mlp(torch.cat(global_inp), dim=1)
        return edge, node, g


class GNResidule(nn.Module):
    """
    Forward model for fixed model.
    The graph is always predefined.
    """
    def __init__(self, state_dim, action_dim, node_attr, edge_attr, graph, layers=3, mid_channels=256):
        super(GNResidule, self).__init__()
        if isinstance(node_attr, np.ndarray):
            node_attr = torch.Tensor(node_attr)
        if isinstance(edge_attr, np.ndarray):
            edge_attr = torch.Tensor(edge_attr)
        if isinstance(graph, np.ndarray):
            graph = torch.LongTensor(graph)
        self.node_attr = nn.Parameter(node_attr, requires_grad=False)

        # graph is a directed graph, generated by a dfs algorithm according to the robot arm's convention.
        # add direction as feature and repeat the edge
        edge_attr = torch.cat((edge_attr, edge_attr[:, -1:]*0+1), dim=1) # add sign
        edge_attr = torch.cat((edge_attr, edge_attr), dim=0)
        edge_attr[edge_attr.shape[0]//2, -1] *= -1
        self.edge_attr = nn.Parameter(edge_attr, requires_grad=False)

        # repeat all edge index
        graph = torch.cat((graph, graph), dim=1)
        self.graph = nn.Parameter(graph, requires_grad=False)

        self.state_dim = state_dim
        self.action_dim = action_dim

        node_dim = state_dim + self.node_attr.shape[1]
        edge_dim = action_dim + self.edge_attr.shape[1]


        self.gn1 = GNBlock(node_dim, edge_dim, layers, mid_channels,
                           output_node_channels=mid_channels,
                           output_edge_channels=mid_channels) # we don't use global now
        self.gn2 = GNBlock(node_dim + mid_channels, edge_dim + mid_channels, layers, mid_channels,
                           output_node_channels=state_dim)

    def build_graph(self, state, action):
        """
        :param state: (batch, n, d_x)
        :param action: (batch, e, action)
        """
        batch_size = state.shape[0]
        node = torch.cat((state, self.node_attr[None,:].expand(batch_size, -1, -1)), dim=2)
        edge = torch.cat((action, action), dim=1) # duplicate actions
        edge = torch.cat((edge, self.edge_attr[None,:].expand(batch_size, -1, -1)), dim=2)

        _, n, d = node.shape
        _, n_e, d_a = edge.shape

        node = node.view(-1, d)
        edge = edge.view(-1, d_a)
        self.n = n

        graph, batch = batch_graphs(batch_size, n, self.graph)
        return node, edge, graph, batch

    def decode(self, node, batch_size):
        return node.view(batch_size, self.n, -1)

    def forward(self, state, action, g=None):
        node, edge, graph, batch = self.build_graph(state, action)
        node, edge, g = self.gn1(node, edge, graph, batch, g)
        node, edge, g = self.gn2(node, edge, graph, batch, g)
        return self.decode(node, len(state))


def add_state(state, delta):
    return torch.cat([
        state[..., :6] + delta[..., :6], # update coordinates
        rot6d.rmul(state[..., 6:12], delta[..., 6:12]),
        rot6d.rmul(state[..., 12:], delta[..., 12:]),
    ], dim=-1)


def del_state(state, target):
    # state - target, in
    return torch.cat([
        state[..., :6] - target[..., :6], # update coordinates
        rot6d.rmul(state[..., 6:12], rot6d.inv(target[..., 6:12])),
        rot6d.rmul(state[..., 12:], rot6d.inv(target[..., 12:]))], dim=-1)


def dist(state, gt, w_x=1., w_w=1.):
    return ((state[...,:6] - gt[...,:6])**2).sum(dim=-1) * w_x +\
           rot6d.rdist(state[..., 6:12], gt[..., 6:12]) * w_w + \
           rot6d.rdist(state[..., 6:12], gt[..., 6:12]) * w_w


def decode_state(state):
    return state[..., :3], state[..., 3:6], state[..., 6:12], state[..., 12:18]


class ForwardAgent(AgentBase):
    def __init__(self, mode, lr, env,*args, **kwargs):
        state_dim = env.state_dim
        action_dim = env.action_dim
        if mode == 'GNN':
            assert env.state_dim == 3 + 3 + 6 + 6
            model = GNResidule(state_dim, action_dim,
                               env.get_node_attr(), env.get_edge_attr(), env.get_graph()
                               *args, **kwargs)
        else:
            kwargs = {
                "num_layer": kwargs['layers'],
                "feature": kwargs["mid_channels"],
                "batch_norm": False,
            }
            if mode == 'mlp':
                n_node = len(env.get_node_attr())
                n_edge = len(env.get_edge_attr())
                model = mlp(state_dim * n_node + action_dim * n_edge, state_dim * n_node,
                            **kwargs)
            else:
                state_dim = env.observation_space.shape[0]
                action_dim = env.action_space.shape[0]
                model = mlp(state_dim + action_dim, state_dim, **kwargs)
            model = Concat(model)

        super(ForwardAgent, self).__init__(model, lr)

        self.forward_model = model
        self.s_norm = Normalizer((state_dim,))
        self.a_norm = Normalizer((action_dim,))
        self.d_norm = Normalizer((state_dim,))

        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, 0.975)
        self.step = 0

    def get_predict(self, s, a, update=True):
        s = self.s_norm(s)
        a = self.a_norm(a)
        output = self.forward_model(s, a)
        output = output.view(s.shape)
        if update:
            return add_state(s, self.d_norm.denorm(output))
        else:
            return output

    def cuda(self):
        self.s_norm.to('cuda:0')
        self.a_norm.to('cuda:0')
        self.d_norm.to('cuda:0')
        return AgentBase.cuda(self)

    def update(self, s, a, t):
        # predict t given s, and a
        delta = del_state(t, s)

        if self.training:
            self.optim.zero_grad()
            self.s_norm.update(s)
            self.a_norm.update(a)
            self.d_norm.update(delta)

        output = self.get_predict(s, a, update=False)
        loss = dist(output, delta).mean()

        if self.training:
            loss.backward()
            self.optim.step()

            self.step += 1
            if self.step % 50000 == 0:
                self.lr_scheduler.step()
        return {
            'loss': tocpu(loss)
        }

