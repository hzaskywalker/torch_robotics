# Baisic GNN forward engine
# The code containts some model similar to ``Graph Networks as Learnable Physics Engines for Inference and Control''
from typing import Any
import torch
import torch.nn as nn
from torch_geometric.utils import scatter_

from robot.utils.models import fc
from robot.utils.normalizer import Normalizer
from robot.utils.trainer import AgentBase
from robot.utils import tocpu

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
    batch = torch.arange(batch_size, device=graph.device)[:, None].expand(-1, n).reshape(-1)
    graph = graph[None, :] + (torch.arange(batch_size, device=graph.device)[:, None, None] * n)

    graph = graph.permute(1, 0, 2).reshape(2, -1)
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

    def __str__(self):
        g = None if self.g is None else self.g.shape
        return f"Node: {self.node.shape}\nEdge: {self.edge.shape}\nGraph: {self.graph}\ng: {g}"


class GNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers, mid_channels, batch_norm=False):
        # in_channels = (node_channels, edge_channels, global_channels)
        super(GNBlock, self).__init__()
        node_channels, edge_channels, global_channels = in_channels
        output_node_channels, output_edge_channels, output_global_channels = out_channels

        if global_channels is None:
            global_channels = 0

        self.edge_feature_mlp = mlp(global_channels + node_channels * 2 + edge_channels,
                            mid_channels, mid_channels, layers, batch_norm=batch_norm)
        self.node_mlp = mlp(global_channels + node_channels + mid_channels,
                            output_node_channels, mid_channels, layers, batch_norm=batch_norm)
        self.output_edge_mlp = fc(mid_channels, output_edge_channels)

        if output_global_channels is not None:
            self.global_mlp = mlp(global_channels + output_node_channels + output_edge_channels,
                                  output_global_channels, mid_channels, layers, batch_norm=batch_norm)
        else:
            self.global_mlp = None

    def forward(self, graph: Graph) -> Graph:
        node = graph.node
        edge = graph.edge
        edge_index = graph.graph
        g = graph.g
        n = node.shape[0]
        batch = graph.batch
        rol, col = edge_index

        edge_inp = [node[rol], node[col], edge]
        if g is not None:
            edge_batch = batch[rol]
            edge_inp.append(g[edge_batch])
        edge_inp = torch.cat(edge_inp, dim=1)
        edge = self.edge_feature_mlp(edge_inp)


        node_inp = [node, scatter_('add', edge, rol, dim=0, dim_size=n)]
        if g is not None:
            node_inp.append(g[batch])
        node_inp = torch.cat(node_inp, dim=1)
        node = self.node_mlp(node_inp)

        if self.global_mlp is not None:
            batch_size = g.shape[0] if g is not None else batch.max() + 1
            global_inp = [scatter_('add', node, batch, dim=0, dim_size=batch_size),
                          scatter_('add', edge, batch[rol], dim=0, dim_size=batch_size)]
            if g is not None:
                global_inp.append(global_inp)
            g = self.global_mlp(torch.cat(global_inp, dim=1))

        edge = self.output_edge_mlp(edge)
        return Graph(node, edge, edge_index, batch, g)


class GNResidule(nn.Module):
    """
    Forward model for fixed model.
    The graph is always predefined.
    """
    def __init__(self, in_channels, oup_channels, layers=3, mid_channels=256, use_global=False):
        super(GNResidule, self).__init__()
        # repeat all edge index
        global_channels = mid_channels if use_global else None
        self.gn1 = GNBlock(in_channels+(None,), (mid_channels, mid_channels, global_channels), layers, mid_channels)
        self.gn2 = GNBlock((in_channels[0] + mid_channels, in_channels[1]+mid_channels, None), oup_channels+(None,), layers, mid_channels)

    def forward(self, graph):
        graph_mid = self.gn1(graph)
        output = self.gn2(graph.cat(graph_mid))
        return output

class GraphNormalizer:
    """
    Normalize the whole graph
    """
    def __init__(self, in_channels):
        self.node = Normalizer((in_channels[0],))
        self.edge = Normalizer((in_channels[1],))
        self.g = None
        if len(in_channels) > 2:
            if in_channels[2] != None:
                self.g = Normalizer((in_channels,))

    def __call__(self, G):
        g = self.g(G.g) if self.g is not None else G.g
        return Graph(self.node(G.node), self.edge(G.edge), G.graph, G.batch, g)

    def cuda(self):
        self.node.cuda()
        self.edge.cuda()
        if self.g is not None:
            self.g.cuda()

    def update_normalizer(self, G):
        self.node.update(G.node)
        self.edge.update(G.edge)
        if self.g is not None:
            self.g.update(G.g)


def build_graph(node, edge, graph):
    batch_size = node.shape[0]
    _, n, d = node.shape
    _, _, d_a = edge.shape
    node = node.reshape(-1, d)
    edge = edge.reshape(-1, d_a)
    graph, batch = batch_graphs(batch_size, n, graph)
    return Graph(node, edge, graph, batch, None)


class GNNForwardAgent(AgentBase):
    def __init__(self, lr, encode_obs, add_state, compute_reward,
                 inp_dim, oup_dim, graph,
                 layers=3, mid_channels=256):

        self.encode_obs = encode_obs
        self.add_state = add_state
        self.graph = graph
        self.compute_reward = compute_reward
        self.inp_norm = GraphNormalizer(inp_dim) # normalize graph

        self.forward_model = GNResidule(inp_dim, oup_dim, layers=layers, mid_channels=mid_channels)

        self.loss = nn.MSELoss()
        super(GNNForwardAgent, self).__init__(self.forward_model, lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, 0.975)
        self.step = 0
        self.device = 'cpu'

    def update(self, obs, a, geom=None):
        # predict t given s, and a
        # support that s is encoded by state_format
        s = obs[:, 0]
        t = obs[:, 1]
        a = a[:, 0]

        if self.training:
            self.optim.zero_grad()

        output = self.get_predict(s, a)
        assert output.shape == t.shape
        loss = self.loss(output, t.detach())

        if self.training:
            loss.backward()
            self.optim.step()

            self.step += 1
            if self.step % 50000 == 0:
                print("LOWERING DOWN THE LEARNING RATE")
                self.lr_scheduler.step()

        return {
            'loss': tocpu(loss),
        }


    def update_normalizer(self, batch, mode):
        assert mode == 'batch'
        node, edge = self.encode_obs(batch[0][:, 0], batch[1][:, 0])
        self.inp_norm.update_normalizer(Graph(node, edge, None))
        #self.inp_norm.node.update(batch[0][:, 0]) # s
        #self.inp_norm.edge.update(batch[1][:, 0]) # a

    def get_predict(self, s, a):
        node, edge = self.encode_obs(s, a)
        graph = build_graph(node, edge, self.graph)
        graph = self.inp_norm(graph)

        delta = self.forward_model(graph)
        output = self.add_state(s, delta.node, delta.edge)
        return output

    def rollout(self, s, a, goal):
        # s (inp_dim)
        # a (pop, T, acts)
        with torch.no_grad():
            reward = 0
            for i in range(a.shape[1]):
                t = self.get_predict(s, a[:, i])
                reward = self.compute_reward(s, a, t, goal) + reward
                s = t
        return reward


    def __call__(self, s, a):
        with torch.no_grad():
            return self.get_predict(s, a)

    def cuda(self):
        self.device = 'cuda:0'
        self.graph = self.graph.cuda()
        self.inp_norm.cuda()
        return super(GNNForwardAgent, self).cuda()
