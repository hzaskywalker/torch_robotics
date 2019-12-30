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

class Graph:
    def __init__(self, node, edge, graph, batch=None, g=None):
        self.node: torch.Tensor = node #(num_node, d_n)
        self.edge: torch.Tensor = edge # (num_edge, d_e)
        self.graph: torch.LongTensor = graph # (2, num_e)
        self.batch: torch.LongTensor = batch #(num_node, )
        self.g = g

    def cat(self, b):
        g = None
        if self.g is None:
            g = b.g
        elif b.g is None:
            g = self.g
        else:
            g = torch.cat((self.g, b.g), dim=1)
        return Graph(torch.cat((self.node, b.node), dim=1), torch.cat((self.edge, b.edge), dim=1), self.graph, self.batch, g)

    def __add__(self, other):
        g = None if self.g is None else self.g + other.g
        return Graph(self.node + other.node, self.edge + other.edge, self.graph, self.batch, g)


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

    def forward(self, graph: Graph) -> Graph:
        node = graph.node
        edge = graph.edge
        edge_index = graph.graph
        g = graph.g
        batch = graph.batch
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

        return Graph(node, edge, edge_index, batch, g)


class GNResidule(nn.Module):
    """
    Forward model for fixed model.
    The graph is always predefined.
    """
    def __init__(self, node_dim, edge_dim, output_node_dim, output_edge_dim, layers=3, mid_channels=256):
        super(GNResidule, self).__init__()
        # repeat all edge index
        self.gn1 = GNBlock(node_dim, edge_dim, layers, mid_channels,
                           output_node_channels=mid_channels,
                           output_edge_channels=mid_channels) # we don't use global now
        self.gn2 = GNBlock(node_dim + mid_channels, edge_dim + mid_channels, layers, mid_channels,
                           output_node_channels=output_node_dim, output_edge_channels=output_edge_dim)

    def forward(self, graph):
        graph_mid = self.gn1(graph)
        output = self.gn2(graph.cat(graph_mid))
        return output

# 先不加normalizer 试试吧。
#class GraphNormalizer:
#    def __init__(self, state_dim, action_dim, g_dim=None):
#        pass


class GNNForwardAgent(AgentBase):
    def __init__(self, lr, env, *args, **kwargs):
        self.init_graph(env)
        self.state_format = env.state_format

        node_dim = self.state_format.d + self.node_attr.shape[-1]
        edge_dim = env.observation_space.shape[-1] + self.edge_attr.shape[-1]

        assert env.state_dim == 3 + 3 + 6 + 6
        model = GNResidule(node_dim, edge_dim, node_dim, 1, *args, **kwargs)

        super(GNNForwardAgent, self).__init__(model, lr)

        self.forward_model = model
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, 0.975)
        self.step = 0


    def init_graph(self, env):
        graph = env.get_graph()
        node_attr = env.get_node_attr()
        edge_attr = env.get_edge_attr()

        if isinstance(node_attr, np.ndarray):
            node_attr = torch.Tensor(node_attr)
        if isinstance(edge_attr, np.ndarray):
            edge_attr = torch.Tensor(edge_attr)
        if isinstance(graph, np.ndarray):
            graph = torch.LongTensor(graph)

        graph = torch.cat((graph, graph), dim=1)
        self.graph = graph
        self.node_attr = node_attr

        # graph is a directed graph, connect parent to child
        # add direction as feature and repeat the edge

        edge_attr = torch.cat((edge_attr, edge_attr[:, -1:]*0+1), dim=1) # add sign
        edge_attr = torch.cat((edge_attr, edge_attr), dim=0)
        edge_attr[edge_attr.shape[0]//2, -1] *= -1
        self.edge_attr = edge_attr
        self.action_list = torch.LongTensor(env.action_to_edge())


    def build_graph(self, state, action, g=None):
        """
        :param state: (batch, n, d_x)
        :param action: (batch, e, action)
        """
        batch_size = state.shape[0]
        node = torch.cat((state, self.node_attr[None,:].expand(batch_size, -1, -1)), dim=2)

        tmp = torch.zeros((action.shape[0], self.edge_attr.shape[0]//2, action.shape[-1])) # only half
        tmp[:, self.action_list] = action
        edge = torch.cat((tmp, tmp), dim=1) # duplicate actions
        edge = torch.cat((edge, self.edge_attr[None,:].expand(batch_size, -1, -1)), dim=2)

        _, n, d = node.shape
        _, n_e, d_a = edge.shape

        self.n = n

        # flatten the graph
        node = node.view(-1, d)
        edge = edge.view(-1, d_a)
        graph, batch = batch_graphs(batch_size, n, self.graph)
        return Graph(node, edge, graph, batch, g)


    def update(self, s, a, t):
        # predict t given s, and a
        # support that s is encoded by state_format

        if self.training:
            self.optim.zero_grad()

        s_node, _ = self.state_format.decode(s)
        t_node, _ = self.state_format.decode(t)

        graph = self.build_graph(s_node, a)
        output = self.forward_model(graph)

        delta = self.state_format.delete(t_node, s_node)
        loss = self.state_format.dist(output.node, delta).mean()

        if self.training:
            loss.backward()
            self.optim.step()

            self.step += 1
            if self.step % 50000 == 0:
                self.lr_scheduler.step()
        return {
            'loss': tocpu(loss)
        }

