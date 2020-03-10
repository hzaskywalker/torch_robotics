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
    def __init__(self, node_channels, edge_channels, layers, mid_channels, global_channels=None,
                 output_node_channels=None, output_edge_channels=None, output_global_channels=None):
        super(GNBlock, self).__init__()
        if global_channels is None:
            global_channels = 0

        self.edge_feature_mlp = mlp(global_channels + node_channels * 2 + edge_channels,
                            mid_channels, mid_channels, layers, batch_norm=False)

        self.node_mlp = mlp(global_channels + node_channels + mid_channels,
                            output_node_channels, mid_channels, layers, batch_norm=False)
        self.output_edge_mlp = fc(mid_channels, output_edge_channels)

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
    def __init__(self, node_dim, edge_dim, output_node_dim, output_edge_dim, layers=3, mid_channels=256, use_global=False):
        super(GNResidule, self).__init__()
        # repeat all edge index
        global_channels = mid_channels if use_global else None
        self.gn1 = GNBlock(node_dim, edge_dim, layers, mid_channels,
                           output_node_channels=mid_channels,
                           output_edge_channels=mid_channels,
                           output_global_channels=global_channels) # we don't use global now
        self.gn2 = GNBlock(node_dim + mid_channels, edge_dim + mid_channels, layers, mid_channels,
                           global_channels=global_channels,
                           output_node_channels=output_node_dim,
                           output_edge_channels=output_edge_dim,
                           output_global_channels=None)

    def forward(self, graph):
        graph_mid = self.gn1(graph)
        output = self.gn2(graph.cat(graph_mid))
        return output

class GraphNormalizer(nn.Module):
    """
    Normalize the whole graph
    """
    def __init__(self, state_dim, edge_dim, g_dim=None):
        super(GraphNormalizer, self).__init__()
        self.node: Normalizer = Normalizer((state_dim,))
        self.edge: Normalizer = Normalizer((edge_dim,))
        if g_dim is not None:
            self.g = Normalizer((g_dim,))
        else:
            self.g = None

    def __call__(self, G):
        g = self.g(G.g) if self.g is not None else G.g
        return Graph(self.node(G.node), self.edge(G.edge), G.graph, G.batch, g)

    def update(self, G):
        self.node.update(G.node)
        self.edge.update(G.edge)
        if self.g is not None:
            self.g.update(G.g)


class ForwardModel(nn.Module):
    def __init__(self, inp_dim, oup_dim, attr_dim, graph, layers, mid_channels):
        super(ForwardModel, self).__init__()
        # TODO: better way to get the number of nodes..
        n = graph.max() + 1
        self.node_attr = nn.Parameter(torch.randn(size=(n, attr_dim[0])), requires_grad=True)
        self.edge_attr = nn.Parameter(torch.randn(size=(n, attr_dim[1])), requires_grad=True)
        self.graph = nn.Parameter(graph, requires_grad=False)

        node_dim = inp_dim[0] + attr_dim[0] # node dim
        edge_dim = inp_dim[1] + attr_dim[1]
        self.model = GNResidule(node_dim, edge_dim, oup_dim[0], oup_dim[1], layers=layers, mid_channels=mid_channels)

    def build_graph(self, node, edge):
        batch_size = node.shape[0]
        node = torch.cat((node, self.node_attr[None, :].expand(batch_size, -1, -1)), dim=2)
        edge = torch.cat((edge, self.edge_attr[None, :].expand(batch_size, -1, -1)), dim=2)

        _, n, d = node.shape
        _, _, d_a = edge.shape
        node = node.view(-1, d)
        edge = edge.view(-1, d_a)
        self.n = n
        graph, batch = batch_graphs(batch_size, n, self.graph)
        return Graph(node, edge, graph, batch, None)

    def forward(self, node, edge):
        graph = self.build_graph(node, edge)
        delta = self.model(graph)
        return delta


class GNNForwardAgent(AgentBase):
    def __init__(self, lr, encode_obs, add_state, compute_reward,
                 inp_dim, oup_dim, attr_dim, graph,
                 layers=3, mid_channels=256):

        self.attr_dim = attr_dim
        self.graph = graph
        self.encode_obs = encode_obs

        self.add_state = add_state
        self.compute_reward = compute_reward

        self.inp_norm = GraphNormalizer(inp_dim[0], inp_dim[1]) # normalize graph
        self.oup_norm = GraphNormalizer(oup_dim[0], oup_dim[1])
        model = ForwardModel(inp_dim, oup_dim, attr_dim, layers, mid_channels)
        super(GNNForwardAgent, self).__init__(model, lr)

        self.forward_model = model
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, 0.975)
        self.step = 0


    def update(self, obs, a, geom=None):
        # predict t given s, and a
        # support that s is encoded by state_format
        s = obs[:, 0]
        t = obs[:, 1]

        # now let's begin...
        # it's a problem that if geom is given as the input
        # our answer is, it depends....
        #     the thing is that we now have a different way to represent one things... should we use all of the informations?
        #     Actually I think so in some sense.
        #     But better way is to calculate the Jacobian and Hamiltonian.. with the neural network... what's that?

        if self.training:
            self.optim.zero_grad()

        delta = self.extension.sub(t, s)

        if self.inp_norm is not None:
            graph = self.inp_norm(graph)
        if self.oup_norm is not None:
            delta = self.oup_norm(delta)

        output = self.decode_node(self.forward_model(graph))
        loss = self.extension.distance(output, delta).mean()

        if self.training:
            if self.inp_norm is not None:
                self.inp_norm.update(graph)
            if self.oup_norm is not None:
                self.oup_norm.update(delta)

            loss.backward()
            self.optim.step()

            self.step += 1
            if self.step % 50000 == 0:
                self.lr_scheduler.step()

        return {
            'loss': tocpu(loss),
            'st_distance': tocpu(self.extension.distance(s, t).mean()),
        }

    def get_predict(self, s, a):
        graph = self.build_graph(s, a)
        if self.inp_norm is not None:
            graph = self.inp_norm(graph)

        if self.oup_norm is not None:
            delta = self.oup_norm.denorm(delta)
        out = self.extension.add(s, delta)
        return out

    def visualize(self, prefix, data, dic, env, **kwargs):
        s, a, t = data
        predict = self.get_predict(s, a)
        t_node = t

        imgs = []
        idx = 0
        for a, b in zip(t_node, predict):
            imgs.append(
                np.concatenate((env.render_state(tocpu(a)), env.render_state(tocpu(b))), axis=1)
            )
            idx += 1
            if idx > 10:
                break
        img = np.array(imgs)
        dic['{}_img'.format(prefix)] = img

    def __call__(self, s, a):
        with torch.no_grad():
            return self.get_predict(s, a)

    def rollout(self, s, a):
        for i in range(a.shape[1]):
            s = self(s, a[:, i])
        return s
