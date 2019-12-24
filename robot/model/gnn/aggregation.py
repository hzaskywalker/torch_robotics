#!/usr/bin/env python
# coding=utf-8

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import softmax
from torch_geometric.data import Data
from torch_scatter import scatter_max
from torch_scatter import scatter_add

import torch
import torch.nn as nn
import numpy as np

from .utils import cal_size_list, MLP

def attention_fn(x, scores, index, size=None):
    weights = softmax(scores.squeeze(), index, size)
    return scatter_add(x*weights.unsqueeze(-1), index, dim=0, dim_size=size)

class AttentionAggregation(nn.Module):
    def __init__(self, score_module, embedding_module, *args, **kwargs):
        super(AttentionAggregation, self).__init__()
        self.score_module = score_module
        self.embedding_module = embedding_module
    def forward(self, x, attention_x, index, size, *args ,**kwargs):
        x = self.embedding_module(x)
        scores = self.score_module(attention_x)
        return attention_fn(x, scores, index, size)

class _Aggregation(nn.Module):
    def __init__(self,*args,**kwargs):
        super(_Aggregation,self).__init__()
    def forward(self, *args, **kwargs):
        raise NotImplementedError

class MaxAggregation(_Aggregation):
    def forward(self, x, index, size, *args, **kwargs):
        return global_max_pool(x, index, size=size)

class MinAggregation(_Aggregation):
    def forward(self, x, index, size, *args, **kwargs):
        return -global_max_pool(-x, index, size=size)

class MeanAggregation(_Aggregation):
    def forward(self,x, index, size, *args, **kwargs):
        return global_mean_pool(x, index, size=size)

class SumAggregation(_Aggregation):
    def forward(self,x, index, size, *args, **kwargs):
        return global_add_pool(x, index, size=size)

class IdentityAggregation(_Aggregation):
    def forward(self,x, index, size, *args, **kwargs):
        return x

class KTopAggregation(_Aggregation):
    def __init__(self, in_channels, hidden_size=64,
                 pooling_score_layer_num=2,
                 activation_name='LeakyReLU',
                 k=3,
                 *args, **kwargs):
        super(KTopAggregation, self).__init__(*args, **kwargs)
        self.k = k

        activation_type = globals()[activation_name]
        embedding_dim_list = np.concatenate([
            cal_size_list(in_channels, hidden_size, int(pooling_score_layer_num/2)),
            cal_size_list(hidden_size, self.k, pooling_score_layer_num-int(pooling_score_layer_num/2))[1:]
        ], axis=0)
        self.embedding_module = MLP(embedding_dim_list, activation=activation_type, last_activation=nn.Identity)
        head_dim_list = cal_size_list(in_channels*self.k, in_channels, 1)
        self.head_module = MLP(head_dim_list, activation=activation_type, )
    def forward(self, data, analyse_flag=False):
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

class GenAttentionAggregation(AttentionAggregation):
    def __init__(self, score_module, embedding_module, size_update_module, *args, **kwargs):
        super(GenAttentionAggregation, self).__init__(score_module, embedding_module)
        self.size_update_module = size_update_module
    def forward(self, x, attention_x, index, size, *args, **kwargs):
        mean = super(GenAttentionAggregation, self).forward(x, attention_x, index, size, *args, **kwargs)
        sizes = global_add_pool(torch.ones([x.size(0),1], dtype=x.dtype, device=x.device),
                                index, size=size)
        updated_sizes = self.size_update_module(sizes)
        return mean * updated_sizes


def get_readout_func(name):
    return globals().get('%sAggregation'%name.lower().capitalize())()
