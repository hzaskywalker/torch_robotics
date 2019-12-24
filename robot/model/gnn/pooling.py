#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch_geometric.nn as gnn
from torch.nn import LeakyReLU
from torch_scatter import scatter_max
from torch_geometric.data import Data
from torch_geometric.utils import softmax
from torch_scatter import scatter_add

import numpy as np

from .utils import cal_size_list, MLP
from .utils import get_build_edges_func

class _Pooling(nn.Module):
    def __init__(self, *args, **kwargs):
        super(_Pooling, self).__init__()
    def forward(self, data):
        raise NotImplementedError
    def correlation_loss(self, data, correlation_name):
        raise NotImplementedError

def filter_adj(edge_index, edge_attr, perm, num_nodes):
    mask = perm.new_full((num_nodes, ), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i

    row, col = edge_index
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return torch.stack([row, col], dim=0), edge_attr

class TopKPooling(_Pooling):
    def __init__(self, in_channels, hidden_size=64,
                 pooling_score_layer_num=2,
                 activation_name='LeakyReLU',
                 k=3,
                 *args, **kwargs):
        super(TopKPooling, self).__init__(*args, **kwargs)
        self.k = k

        activation_type = globals()[activation_name]
        embedding_dim_list = np.concatenate([
            cal_size_list(in_channels, hidden_size, int(pooling_score_layer_num/2)),
            cal_size_list(hidden_size, 1, pooling_score_layer_num-int(pooling_score_layer_num/2))[1:]
        ], axis=0)
        self.embedding_module = MLP(embedding_dim_list, activation=activation_type, last_activation=nn.Identity)
    def forward(self, data, analyse_flag=False, *args, **kwargs):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if 'edge_attr' in data.__dict__ else None
        num_graphs = data.num_graphs

        score_value = self.embedding_module(x)
        score_value = softmax(score_value, batch)
        score = score_value.clone().detach().reshape([-1])
        score_inf = torch.min(score) - 1
        perm = []
        for _ in range(self.k):
            perm.append(scatter_max(score, batch, dim=0)[-1])
            score[perm[-1]] = score_inf
        perm = torch.cat(perm)

        x, batch = x[perm] * score_value[perm], batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm, score.size(0))

        output = Data(
            x = x, edge_index=edge_index,
            batch=batch, edge_attr=edge_attr,
            num_graphs=num_graphs,
        )
        if analyse_flag:
            output = (output, perm)
        return output

class SoftmaxPooling(_Pooling):
    def __init__(self, in_channels, hidden_size=64,
                 pooling_score_layer_num=2,
                 activation_name='LeakyReLU',
                 category_num=3,
                 *args, **kwargs):
        super(SoftmaxPooling, self).__init__(*args, **kwargs)
        self.category_num = category_num

        activation_type = globals()[activation_name]
        score_dim_list = np.concatenate([
            cal_size_list(in_channels, hidden_size, pooling_score_layer_num-int(pooling_score_layer_num/2)),
            cal_size_list(hidden_size, category_num, int(pooling_score_layer_num/2))[1:]
        ], axis=0)
        self.score_module = MLP(score_dim_list, activation=activation_type, last_activation=nn.Identity)
        embedding_dim_list = np.concatenate([
            cal_size_list(in_channels, hidden_size, pooling_score_layer_num-int(pooling_score_layer_num/2)),
            cal_size_list(hidden_size, in_channels-category_num, int(pooling_score_layer_num/2))[1:]
        ], axis=0)
        self.embedding_module = MLP(embedding_dim_list, activation=activation_type, last_activation=nn.Identity)
    def forward(self, data, *args, **kwargs):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if 'edge_attr' in data.__dict__ else None
        num_graphs = data.num_graphs
        context = self.embedding_module(x)

        score_value = self.score_module(x)
        score_value = F.softmax(score_value, dim=-1)
        score_cat = torch.argmax(score_value, dim=-1)
        perm = torch.arange(score_value.size(0), dtype=torch.long, device=x.device)[score_cat==0]

        x = torch.cat([score_value, context], dim=-1)[perm]
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm, score_value.size(0))

        return Data(
            x = x, edge_index=edge_index,
            batch=batch, edge_attr=edge_attr,
            num_graphs=num_graphs,
        )

class SoftmaxV2Pooling(SoftmaxPooling):
    def forward(self, data, *args, **kwargs):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if 'edge_attr' in data.__dict__ else None
        num_graphs = data.num_graphs
        context = self.embedding_module(x)

        score_value = self.score_module(x)
        score_value = F.softmax(score_value, dim=-1)
        score_cat = torch.argmax(score_value, dim=-1)
        perm = torch.arange(score_value.size(0), dtype=torch.long, device=x.device)[score_cat!=0]

        x = torch.cat([score_value, context], dim=-1)[perm]
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm, score_value.size(0))

        return Data(
            x = x, edge_index=edge_index,
            batch=batch, edge_attr=edge_attr,
            num_graphs=num_graphs,
        )

class TopKGraphPooling(TopKPooling):
    def __init__(self, in_channels, hidden_size=64,
                 pooling_score_layer_num=2,
                 activation_name='LeakyReLU',
                 k=3,
                 graph_name='fully',
                 *args, **kwargs):
        super(TopKGraphPooling, self).__init__(in_channels, hidden_size,
                                               pooling_score_layer_num,
                                               activation_name, k, graph_name,
                                               *args, **kwargs)
        self.build_edges_func = get_build_edges_func(graph_name)
    def forward(self, data, *args, **kwargs):
        data = super(TopKGraphPooling, self).forward(data)
        x, batch, num_graphs, device = data.x, data.batch, data.num_graphs, data.x.device
        num_nodes = scatter_add(x.new_ones(x.size(0)), batch, dim_size=num_graphs)
        edge_index_list, edge_attr_list = list(zip(*(
            self.build_edges_func(int(N), device) for N in num_nodes.detach().cpu().numpy()
        )))
        cum_num_nodes = [0]+torch.cumsum(num_nodes,dim=0).detach().cpu().numpy().tolist()[:-1]
        edge_index_list = [
            cnum + edge_index
            for cnum, edge_index in zip(cum_num_nodes, edge_index_list)
        ]

        edge_index = torch.cat(edge_index_list, dim=-1)
        edge_attr = torch.cat(edge_attr_list, dim=0)

        return Data(
            x = x,
            batch = batch,
            edge_index = edge_index,
            edge_attr = edge_attr,
            y = data.y,
            num_graphs = num_graphs,
        )

class IdentityPooling(_Pooling):
    def __init__(self, *args, **kwargs):
        super(IdentityPooling, self).__init__(*args, **kwargs)
    def forward(self, data, *args, **kwargs):
        return data

def correlation_loss(score, name):
    if 'softmax' in name:
        score = F.softmax(score, dim=0)
    corr = torch.mm(score.t(), score)
    if 'cross' in name:
        corr = corr * (torch.ones_like(corr)-torch.eye(score.size(-1), dtype=score.dtype, device=score.device))
    return torch.mean(corr)

'''
actually wrong name, should be KTopReadout
exist here for historical reasons
don't change or use it....
'''
class KTopPooling(_Pooling):
    def __init__(self, in_channels, hidden_size=64,
                 pooling_score_layer_num=2,
                 activation_name='LeakyReLU',
                 k=3,
                 *args, **kwargs):
        super(KTopPooling, self).__init__(*args, **kwargs)
        self.k = k

        activation_type = globals()[activation_name]
        embedding_dim_list = np.concatenate([
            cal_size_list(in_channels, hidden_size, int(pooling_score_layer_num/2)),
            cal_size_list(hidden_size, self.k, pooling_score_layer_num-int(pooling_score_layer_num/2))[1:]
        ], axis=0)
        self.embedding_module = MLP(embedding_dim_list, activation=activation_type, last_activation=nn.Identity)
        head_dim_list = cal_size_list(in_channels*self.k, in_channels, 1)
        self.head_module = MLP(head_dim_list, activation=activation_type, )
    def forward(self, data, analyse_flag=False, *args, **kwargs):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if 'edge_attr' in data.__dict__ else None
        num_graphs = data.num_graphs

        score_values = self.embedding_module(x)
        perm,  x_list = [], []
        for kk in range(self.k):
            score_value = softmax(score_values[:,kk], batch)
            score = score_value.clone().detach().reshape([-1])
            index = scatter_max(score, batch, dim=0)[-1]
            perm.append(index)
            x_list.append(x[index]*score_value[index].unsqueeze(-1))
        x = torch.stack(x_list, dim=-2).reshape([x_list[0].size(0),-1])

        x = self.head_module(x)
        batch = batch[perm[0]]
        edge_index = torch.zeros([0, 2], dtype=torch.long, device=x.device)
        edge_attr = torch.zeros([0, 1], dtype=torch.float, device=x.device)

        perm = torch.cat(perm)

        output = Data(
            x = x, edge_index=edge_index,
            batch=batch, edge_attr=edge_attr,
            num_graphs=num_graphs,
        )
        if analyse_flag:
            output = (output, perm)
        return output
    def correlation_loss(self, data, correlation_name='simple'):
        score_values = self.embedding_module(data.x)
        return correlation_loss(score_values, correlation_name)

class KTopNewPooling(_Pooling):
    def __init__(self, in_channels, hidden_size=64,
                 pooling_score_layer_num=2,
                 activation_name='LeakyReLU',
                 k=3,
                 *args, **kwargs):
        super(KTopNewPooling, self).__init__(*args, **kwargs)
        self.k = k

        activation_type = globals()[activation_name]
        embedding_dim_list = np.concatenate([
            cal_size_list(in_channels, hidden_size, int(pooling_score_layer_num/2)),
            cal_size_list(hidden_size, self.k, pooling_score_layer_num-int(pooling_score_layer_num/2))[1:]
        ], axis=0)
        self.embedding_module = MLP(embedding_dim_list, activation=activation_type, last_activation=nn.Identity)
        head_dim_list = cal_size_list(in_channels+self.k, in_channels, 1)
        self.head_module = MLP(head_dim_list, activation=activation_type, )
    def forward(self, data, analyse_flag=False, *args, **kwargs):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if 'edge_attr' in data.__dict__ else None
        num_graphs = data.num_graphs

        score_values = self.embedding_module(x)
        perm,  x_list = [], []
        for kk in range(self.k):
            score_value = softmax(score_values[:,kk], batch)
            score = score_value.clone().detach().reshape([-1])
            index = scatter_max(score, batch, dim=0)[-1]
            perm.append(index)
            x_list.append(torch.cat([
                x[index]*score_value[index].unsqueeze(-1),
                torch.stack([torch.eye(self.k, dtype=x.dtype, device=x.device)[kk]]*x[index].size(0), dim=0),
            ], dim=-1))
        x, perm = torch.cat(x_list, dim=0), torch.cat(perm)
        x = self.head_module(x)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm, score.size(0))

        output = Data(
            x = x, edge_index=edge_index,
            batch=batch, edge_attr=edge_attr,
            num_graphs=num_graphs,
        )
        if analyse_flag:
            output = (output, perm)
        return output
    def correlation_loss(self, data, correlation_name='simple'):
        score_values = self.embedding_module(data.x)
        return correlation_loss(score_values, correlation_name)

class KTopNewGraphPooling(KTopNewPooling):
    def __init__(self, in_channels, hidden_size=64,
                 pooling_score_layer_num=2,
                 activation_name='LeakyReLU',
                 k=3, correlation_name='simple',
                 graph_name='fully',
                 *args, **kwargs):
        super(KTopNewGraphPooling, self).__init__(in_channels, hidden_size,
                                               pooling_score_layer_num,
                                               activation_name, k, graph_name,
                                               *args, **kwargs)
        self.build_edges_func = get_build_edges_func(graph_name)
    def forward(self, data, analyse_flag=False, *args, **kwargs):
        out = super(KTopNewGraphPooling, self).forward(data, analyse_flag)
        if not analyse_flag:
            data = out
        else:
            data, perm = out
        x, batch, num_graphs, device = data.x, data.batch, data.num_graphs, data.x.device
        num_nodes = scatter_add(x.new_ones(x.size(0)), batch, dim_size=num_graphs)
        edge_index_list, edge_attr_list = list(zip(*(
            self.build_edges_func(int(N), device) for N in num_nodes.detach().cpu().numpy()
        )))
        cum_num_nodes = [0]+torch.cumsum(num_nodes,dim=0).detach().cpu().numpy().tolist()[:-1]
        edge_index_list = [
            cnum + edge_index
            for cnum, edge_index in zip(cum_num_nodes, edge_index_list)
        ]

        edge_index = torch.cat(edge_index_list, dim=-1)
        edge_attr = torch.cat(edge_attr_list, dim=0)

        out = Data(
            x = x,
            batch = batch,
            edge_index = edge_index,
            edge_attr = edge_attr,
            y = data.y,
            num_graphs = num_graphs,
        )
        if analyse_flag:
            out = (out, perm)
        return out
