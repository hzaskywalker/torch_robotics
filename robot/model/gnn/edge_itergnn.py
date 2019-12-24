#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .itergnn import IdentityUpdate, MaxUpdate, MeanUpdate, AddUpdate, GateUpdate
from .utils import cal_size_list, MLP
from .aggregation import *

class EdgeBody(nn.Module):
    def __init__(self, edge_embedding_module, aggregation_module, update_module,
                 readout_module, score_module):
        super(EdgeBody, self).__init__()
        self.edge_embedding_module = edge_embedding_module
        self.aggregation_module = aggregation_module
        self.update_module = update_module
        self.readout_module = readout_module
        self.score_module = score_module
    def forward(self, *args, **kwargs):
        x = kwargs.get('x')
        edge_feat = kwargs.get('hidden_edge_feat')
        edge_index = kwargs.get('edge_index')
        edge_attr = kwargs.get('edge_attr')
        batch = kwargs.get('batch')
        num_nodes = x.size(0)
        num_graphs = kwargs.get('num_graphs')
        # print('Body/Input:',x.size(), node_feat.size(), edge_index.size(), edge_attr.size())

        rol, col = edge_index
        edge_att_feat = torch.cat([x[rol], x[col],
                                   edge_attr,
                                   edge_feat], dim=-1)
        node_feat = self.aggregation_module(x=edge_feat,
                                            attention_x=edge_att_feat,
                                            index=col,
                                            size=num_nodes)
        edge_feat_cand = self.edge_embedding_module(
            torch.cat([x[rol], node_feat[rol],
                       x[col], node_feat[col],
                       edge_attr], dim=-1)
        )

        # print('Body/Edge_embedded:',node_feat.size(), node_feat_cand.size(), x.size())
        update_att_feat = torch.cat([x[rol], x[col],
                                     edge_attr,
                                     edge_feat,
                                     edge_feat_cand], dim=-1)
        edge_feat = self.update_module(x=edge_feat,
                                       cand_x = edge_feat_cand,
                                       attention_x = update_att_feat)
        edge_feat_att = torch.cat([x[rol], x[col],
                                   edge_feat, edge_attr], dim=-1)
        graph_feat = self.readout_module(x=edge_feat,
                                         attention_x=edge_feat_att,
                                         index=batch[rol],
                                         size=num_graphs,)
        confidence = torch.sigmoid(self.score_module(graph_feat))
        return edge_feat, confidence

class IterGNN(nn.Module):
    def __init__(self, embedding_module, body_module, readout_module, head_module, max_iter=30, *args, **kwargs):
        super(IterGNN, self).__init__()
        self.embedding_module = embedding_module
        self.body_module = body_module
        self.readout_module = readout_module
        self.head_module = head_module
        self.max_iter = max_iter
    def forward(self, data, output_edge_feat=False, layer_num_flag=False, *args, **kwargs):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        num_graphs = data.num_graphs
        rol, col = edge_index

        hidden_edge_feat = self.embedding_module(edge_attr)
        left_confidence = torch.ones([edge_attr.size(0),1], dtype=x.dtype, device=x.device)
        for iter_num in range(self.max_iter):
            if torch.max(left_confidence).item() > 1e-7:
                current_hidden_edge_feat, current_confidence = self.body_module(
                    x = x, edge_index = edge_index, edge_attr=edge_attr, batch=batch,
                    hidden_edge_feat=hidden_edge_feat, left_confidence=left_confidence,
                    num_graphs=num_graphs,
                )
                hidden_edge_feat = hidden_edge_feat + left_confidence*current_confidence[batch[rol]]*current_hidden_edge_feat
                left_confidence = left_confidence*(1.-current_confidence[batch[rol]])

        attention_x = torch.cat([x[rol], x[col],
                                 hidden_edge_feat, edge_attr], dim=-1)
        graph_feat = self.readout_module(
            x=hidden_edge_feat, attention_x=attention_x
            , index=batch[rol], size=num_graphs
        )

        out = self.head_module(graph_feat)

        output = out, scatter_add(x.new_ones(x.size(0)), batch, dim_size=num_graphs).reshape([-1,1])
        if output_edge_feat:
            output = output + (hidden_edge_feat, )
        if layer_num_flag:
            output = output + (iter_num, )
        return output

def EdgeBaseline(in_channels, edge_channels, out_channels,
             net_name='IterGNN', max_iter=30,
             hidden_size=64, embedding_layer_num=2,
             edge_embedding_layer_num=2,
             aggregation_name='Max', aggregation_score_layer_num=1, aggregation_size_layer_num=2,
             update_module_name='Max', update_gate_layer_num=1,
             readout_name='Attention', readout_score_layer_num=1, readout_size_layer_num=2,
             confidence_layer_num=1,
             head_layer_num=1,
             last_bias=True,
             *args, **kwargs,
             ):

    embedding_size_list = cal_size_list(edge_channels, hidden_size, embedding_layer_num)
    embedding_module = MLP(embedding_size_list, last_bias=last_bias)

    edge_embedding_size_list = cal_size_list(
        2*in_channels+edge_channels+2*(in_channels if embedding_layer_num == 0 else hidden_size)
        , hidden_size, edge_embedding_layer_num
    )
    edge_embedding_module = MLP(edge_embedding_size_list, last_bias=last_bias)

    aggregation_score_size_list = cal_size_list(
        edge_channels+in_channels*2+(in_channels if embedding_layer_num == 0 else hidden_size),
        1, aggregation_score_layer_num,
    )
    aggregation_score_module = MLP(aggregation_score_size_list, nn.Identity, last_bias=False)
    aggregation_embedding_module = nn.Identity()
    aggregation_size_size_list = [1]+[hidden_size]*(aggregation_size_layer_num-1)+[1]
    aggregation_size_module = MLP(aggregation_size_size_list, last_bias=last_bias)
    aggregation_module = globals().get(aggregation_name+'Aggregation')(
        score_module = aggregation_score_module,
        embedding_module = aggregation_embedding_module,
        size_update_module=aggregation_size_module,
    )

    update_gate_size_list = cal_size_list(in_channels*2+edge_channels+2*hidden_size, 1, update_gate_layer_num)
    update_gate_module =  MLP(update_gate_size_list, nn.Identity)
    update_update_module = nn.Identity()
    update_module = globals().get(update_module_name+'Update')(
        gate_module=update_gate_module,
        update_module=update_update_module
    )

    readout_score_size_list = cal_size_list(in_channels*2+edge_channels+hidden_size, 1, readout_score_layer_num)
    readout_score_module = MLP(readout_score_size_list, nn.Identity, last_bias=False)
    readout_embedding_module = nn.Identity()
    readout_size_size_list = [1]+[hidden_size]*(readout_size_layer_num-1)+[1]
    readout_size_module = MLP(readout_size_size_list, last_bias=last_bias)
    readout_module = globals().get(readout_name+'Aggregation')(
        score_module = readout_score_module,
        embedding_module = readout_embedding_module,
        size_update_module=readout_size_module,
    )

    score_size_list = cal_size_list(hidden_size, 1, confidence_layer_num)
    score_module = MLP(score_size_list, nn.Identity)

    body_module = EdgeBody(
        edge_embedding_module=edge_embedding_module,
        aggregation_module=aggregation_module,
        update_module=update_module,
        readout_module=readout_module,
        score_module=score_module
    )

    head_size_list = cal_size_list(hidden_size, out_channels, head_layer_num)
    head_module = MLP(head_size_list, nn.Identity)

    model_type = globals().get(net_name)
    model = model_type(
        embedding_module = embedding_module,
        body_module = body_module,
        readout_module = readout_module,
        head_module = head_module,
        max_iter = max_iter,
        hidden_size=hidden_size,
    )

    return model
