#!/usr/bin/env python
# coding=utf-8

import math

import torch
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.nn.conv import MessagePassing, GINConv
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_add

from torch.nn import LeakyReLU
from torch_geometric.nn.inits import reset

import numpy as np

class MyGCNConv(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True):
        super(MyGCNConv, self).__init__()
        self.linear = nn.Linear(in_channel, out_channel, bias)
    def forward(self, x, edge_index):
        '''
        x: torch.Tensor, of shape [N,C]
            , where N is the number of nodes, and C is the feature dimension
        edge_index: torch.Tensor, of shape [2,E], of type torch.long
            undirected graph, containing self-loop
        '''
        x = self.linear(x)

        r, c = edge_index
        node_degree = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
        node_degree = node_degree.scatter_add(0, r, torch.ones(r.size(), dtype=x.dtype, device=x.device)) #degree of shape [N,]
        nd_rsqrt = torch.rsqrt(node_degree) #degree ^ -1/2
        nd_rsqrt[nd_rsqrt==float('inf')] = 0
        message = x[c]
        message_norm = message * nd_rsqrt[c].unsqueeze(-1)

        out = torch.zeros_like(x)
        out = out.scatter_add(0, r.unsqueeze(-1).expand([-1, x.size(-1)]), message_norm,)
        out_norm = out * nd_rsqrt.unsqueeze(-1)

        return out_norm

class MyGINConv(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True):
        super(MyGINConv, self).__init__()
        size_list = cal_size_list(in_channel, out_channel, 2)
        self.module = GINConv(MLP(size_list,))
    def forward(self, x, edge_index):
        return self.module(x, edge_index)

class MyGINConvV2(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True):
        super(MyGINConvV2, self).__init__()
        size_list = cal_size_list(in_channel, out_channel, 2)
        self.module = GINConv(MLP(size_list,), train_eps=True)
    def forward(self, x, edge_index):
        return self.module(x, edge_index)

class MeanGINConv(MessagePassing):
    def __init__(self, nn, eps=0, train_eps=False, **kwargs):
        super(MeanGINConv, self).__init__(aggr='mean', **kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, _ = remove_self_loops(edge_index)
        out = self.nn((1 + self.eps) * x + self.propagate(edge_index, x=x))
        return out

    def message(self, x_j):
        return x_j

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

class MyMeanGINConv(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True):
        super(MyMeanGINConv, self).__init__()
        size_list = cal_size_list(in_channel, out_channel, 2)
        self.module = MeanGINConv(MLP(size_list,))
    def forward(self, x, edge_index):
        return self.module(x, edge_index)

class MyMeanGINConvV2(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True):
        super(MyMeanGINConvV2, self).__init__()
        size_list = cal_size_list(in_channel, out_channel, 2)
        self.module = MeanGINConv(MLP(size_list,), train_eps=True)
    def forward(self, x, edge_index):
        return self.module(x, edge_index)

class MyEdgeAttConv(nn.Module):
    def __init__(self, embedding_module, attention_module, update_module, **kwargs):
        super(MyEdgeAttConv, self).__init__()
        self.embedding_module = embedding_module
        self.attention_module = attention_module
        self.update_module = update_module
    def forward(self, x, edge_index):
        '''
        x: torch.Tensor, of shape [N,C]
            , where N is the number of nodes, and C is the feature dimension
        edge_index: torch.Tensor, of shape [2,E], of type torch.long
            undirected graph, containing self-loop
        '''
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        rol, col = edge_index

        embedded_x = self.embedding_module(x)
        attention_weights = self.attention_module(x[rol]-x[col])
        attention_weights = softmax(attention_weights, col, )
        weighted_sum_x = scatter_add(embedded_x[rol]*attention_weights, col, dim=0, dim_size=x.size(0))
        updated_x = self.update_module(weighted_sum_x)

        return updated_x

class ZeroLinear(nn.Linear):
    def reset_parameters(self):
        init.constant_(self.weight, 0.)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

class MySequential(nn.Sequential):
    '''
    More general Sequential with dynamic number of input for forward function
    '''
    def forward(self, *args, **kwargs):
        iterator = iter(self._modules.values())
        module = next(iterator)
        output = module(*args, **kwargs)
        for module in iterator:
            output = module(output, **kwargs)
        return output

def cal_size_list(in_channels, out_channels, layer_num):
    return np.linspace(
        in_channels, out_channels,
        layer_num+1, dtype='int'
    )

def MLP(size_list, last_activation=nn.LeakyReLU, activation=nn.LeakyReLU, last_bias=True, bias_flag=True):
    last_bias = bias_flag and last_bias
    return nn.Sequential(
        *(
            nn.Sequential(nn.Linear(size_list[ln], size_list[ln+1], bias=(bias_flag if ln != len(size_list)-2 else last_bias)),
                           activation() if ln != len(size_list)-2 else last_activation())
            for ln in range(len(size_list)-1)
        )
    )

def get_activation_func(name):
    return globals()[name]

def append_position(img):
    '''
    img is torch.tensor of shape [N,H,W,C]
    '''
    N, H, W, C = img.size()
    dtype, device = img.dtype, img.device

    rol = torch.arange(H, dtype=dtype, device=device).unsqueeze(-1).expand([-1,W])
    rol = rol.unsqueeze(0).expand([N,-1,-1]).unsqueeze(-1)
    col = torch.arange(W, dtype=dtype, device=device).unsqueeze(0).expand([H, -1])
    col = col.unsqueeze(0).expand([N,-1,-1]).unsqueeze(-1)

    return torch.cat([rol, col, img], dim=-1)

def build_fully_connected_edges(N, device):
    index = torch.arange(N, device=device).unsqueeze(-1).expand([-1,N])
    rol = torch.reshape(index, [-1])
    col = torch.reshape(torch.t(index), [-1])
    edge_index = torch.stack([rol, col], dim=0)
    edge_attr = torch.ones([edge_index.size(-1),1], dtype=torch.float, device=device)
    return edge_index, edge_attr

def build_fully_connected_graph(x, y):
    N, device = x.size(0), x.device
    edge_index, edge_attr = build_fully_connected_edges(N, device)
    return Data(
        x = x,
        edge_index = edge_index,
        edge_attr = edge_attr,
        y = y,
    )

def get_build_graph_func(name):
    if name == 'fully':
        return build_fully_connected_graph
    else:
        raise ValueError('Wrong graph name in build_graph_func')

def get_build_edges_func(name):
    if name == 'fully':
        return build_fully_connected_edges
    else:
        raise ValueError('Wrong graph name in build_edges_func')

def main():
    device = torch.device('cpu')

    # conv = MyGCNConv(4,2).to(device)
    # x = torch.arange(1,13, requires_grad=True, dtype=torch.float, device=device).reshape([3,4])
    # edge_index = torch.tensor([[0,1],[1,0],[0,2],[2,0]],dtype=torch.long, device=device).t()
    # out = conv(x, edge_index)

    embedding_module = lambda x:x
    attention_module = lambda x:torch.zeros_like(x)[:,0:1]
    update_module = lambda x:x
    conv = MyEdgeAttConv(embedding_module, attention_module, update_module).to(device)
    x = torch.arange(1,13, requires_grad=True, dtype=torch.float, device=device).reshape([3,4])
    edge_index = torch.tensor([[0,1],[1,2],[2,0]],dtype=torch.long, device=device).t()
    x = conv(x, edge_index)
    x = conv(x, edge_index)
    x = conv(x, edge_index)

if __name__ == '__main__':
    main()
