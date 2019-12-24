#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_scatter import scatter_add
import numpy as np

from torch_geometric.nn import EdgeConv, GCNConv, GATConv
from .utils import MyGINConv as GINConv, MyGINConvV2 as GINConvV2
from .utils import MyMeanGINConv as MeanGINConv, MyMeanGINConvV2 as MeanGINConvV2
from .utils import MySequential

from .utils import cal_size_list, MLP
from .aggregation import *

import copy

class IterGNN(nn.Module):
    def __init__(self, embedding_module, body_module, readout_module, head_module, max_iter=30, *args, **kwargs):
        super(IterGNN, self).__init__()
        self.embedding_module = embedding_module
        self.body_module = body_module
        self.readout_module = readout_module
        self.head_module = head_module
        self.max_iter = max_iter
    def forward(self, data, output_node_feat=False, layer_num_flag=False, *args, **kwargs):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        num_graphs = data.num_graphs
        # print('Input:',x.size(), edge_index.size(), edge_attr.size(), batch.size())

        hidden_node_feat = self.embedding_module(x)
        left_confidence = torch.ones([batch.size(0),1], dtype=x.dtype, device=x.device)
        # print('Embedded:',hidden_node_feat.size(), left_confidence.size())
        for iter_num in range(self.max_iter):
            if torch.max(left_confidence).item() > 1e-7:
                current_hidden_node_feat, current_confidence = self.body_module(
                    x = x, edge_index = edge_index, edge_attr=edge_attr, batch=batch,
                    hidden_node_feat=hidden_node_feat, left_confidence=left_confidence,
                    num_graphs=num_graphs,
                )
                # print('Body%d:'%iter_num,hidden_node_feat.size(), left_confidence.size()
                      # , current_confidence.size(), current_hidden_node_feat.size())
                hidden_node_feat = hidden_node_feat + left_confidence*current_confidence[batch]*current_hidden_node_feat
                left_confidence = left_confidence*(1.-current_confidence[batch])
                # print(hidden_node_feat.size(), left_confidence.size())
            else:
                break

        attention_x = torch.cat([x, hidden_node_feat], dim=-1)
        graph_feat = self.readout_module(
            x=hidden_node_feat, attention_x=attention_x
            , index=batch, size=num_graphs
        )

        out = self.head_module(graph_feat)

        output = out, scatter_add(x.new_ones(x.size(0)), batch, dim_size=num_graphs).reshape([-1,1])
        if output_node_feat:
            output = output + (hidden_node_feat, )
        if layer_num_flag:
            output = output + (iter_num, )
        return output

class DiffGNN(nn.Module):
    def __init__(self, embedding_module, body_module, readout_module, head_module, max_iter=30, *args, **kwargs):
        super(DiffGNN, self).__init__()
        self.embedding_module = embedding_module
        self.body_module_list = nn.ModuleList([copy.deepcopy(body_module) for _ in range(max_iter)])
        self.readout_module = readout_module
        self.head_module = head_module
        self.max_iter = max_iter
    def forward(self, data, output_node_feat=False, layer_num_flag=False, *args, **kwargs):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        num_graphs = data.num_graphs
        # print('Input:',x.size(), edge_index.size(), edge_attr.size(), batch.size())

        hidden_node_feat = self.embedding_module(x)
        left_confidence = torch.ones([batch.size(0),1], dtype=x.dtype, device=x.device)
        # print('Embedded:',hidden_node_feat.size(), left_confidence.size())
        for iter_num in range(self.max_iter):
            if torch.max(left_confidence).item() > 1e-7:
                current_hidden_node_feat, current_confidence = self.body_module_list[iter_num](
                    x = x, edge_index = edge_index, edge_attr=edge_attr, batch=batch,
                    hidden_node_feat=hidden_node_feat, left_confidence=left_confidence,
                    num_graphs=num_graphs,
                )
                # print('Body%d:'%iter_num,hidden_node_feat.size(), left_confidence.size()
                      # , current_confidence.size(), current_hidden_node_feat.size())
                hidden_node_feat = hidden_node_feat + left_confidence*current_confidence[batch]*current_hidden_node_feat
                left_confidence = left_confidence*(1.-current_confidence[batch])
                # print(hidden_node_feat.size(), left_confidence.size())
            else:
                break

        attention_x = torch.cat([x, hidden_node_feat], dim=-1)
        graph_feat = self.readout_module(
            x=hidden_node_feat, attention_x=attention_x
            , index=batch, size=num_graphs
        )

        out = self.head_module(graph_feat)

        output = out, scatter_add(x.new_ones(x.size(0)), batch, dim_size=num_graphs).reshape([-1,1])
        if output_node_feat:
            output = output + (hidden_node_feat, )
        if layer_num_flag:
            output = output + (iter_num, )
        return output

class SimIterGNN(IterGNN):
    def forward(self, data, output_node_feat=False, layer_num_flag=False, *args, **kwargs):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        num_graphs = data.num_graphs
        # print('Input:',x.size(), edge_index.size(), edge_attr.size(), batch.size())

        hidden_node_feat = self.embedding_module(x)
        left_confidence = torch.ones([batch.size(0),1], dtype=x.dtype, device=x.device)
        # print('Embedded:',hidden_node_feat.size(), left_confidence.size())
        for iter_num in range(self.max_iter):
            current_hidden_node_feat, current_confidence = self.body_module(
                x = x, edge_index = edge_index, edge_attr=edge_attr, batch=batch,
                hidden_node_feat=hidden_node_feat, left_confidence=left_confidence,
                num_graphs=num_graphs,
            )
            # print('Body%d:'%iter_num,hidden_node_feat.size(), left_confidence.size()
                    # , current_confidence.size(), current_hidden_node_feat.size())
            # hidden_node_feat = hidden_node_feat + left_confidence*current_confidence[batch]*current_hidden_node_feat
            hidden_node_feat = current_hidden_node_feat
            # left_confidence = left_confidence*(1.-current_confidence[batch])
            left_confidence = 1. - current_confidence[batch]
            # print(hidden_node_feat.size(), left_confidence.size())

        attention_x = torch.cat([x, hidden_node_feat], dim=-1)
        graph_feat = self.readout_module(
            x=hidden_node_feat, attention_x=attention_x
            , index=batch, size=num_graphs
        )

        out = self.head_module(graph_feat)

        output = out, scatter_add(x.new_ones(x.size(0)), batch, dim_size=num_graphs).reshape([-1,1])
        if output_node_feat:
            output = output + (hidden_node_feat, )
        if layer_num_flag:
            output = output + (iter_num, )
        return output

class SimDiffGNN(DiffGNN):
    def forward(self, data, output_node_feat=False, layer_num_flag=False, *args, **kwargs):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        num_graphs = data.num_graphs
        # print('Input:',x.size(), edge_index.size(), edge_attr.size(), batch.size())

        hidden_node_feat = self.embedding_module(x)
        left_confidence = torch.ones([batch.size(0),1], dtype=x.dtype, device=x.device)
        # print('Embedded:',hidden_node_feat.size(), left_confidence.size())
        for iter_num in range(self.max_iter):
            current_hidden_node_feat, current_confidence = self.body_module_list[iter_num](
                x = x, edge_index = edge_index, edge_attr=edge_attr, batch=batch,
                hidden_node_feat=hidden_node_feat, left_confidence=left_confidence,
                num_graphs=num_graphs,
            )
            # print('Body%d:'%iter_num,hidden_node_feat.size(), left_confidence.size()
                    # , current_confidence.size(), current_hidden_node_feat.size())
            # hidden_node_feat = hidden_node_feat + left_confidence*current_confidence[batch]*current_hidden_node_feat
            hidden_node_feat = current_hidden_node_feat
            # left_confidence = left_confidence*(1.-current_confidence[batch])
            left_confidence = 1. - current_confidence[batch]
            # print(hidden_node_feat.size(), left_confidence.size())

        attention_x = torch.cat([x, hidden_node_feat], dim=-1)
        graph_feat = self.readout_module(
            x=hidden_node_feat, attention_x=attention_x
            , index=batch, size=num_graphs
        )

        out = self.head_module(graph_feat)

        output = out, scatter_add(x.new_ones(x.size(0)), batch, dim_size=num_graphs).reshape([-1,1])
        if output_node_feat:
            output = output + (hidden_node_feat, )
        if layer_num_flag:
            output = output + (iter_num, )
        return output

class SigmoidIterGNN(IterGNN):
    def forward(self, data, output_node_feat=False, layer_num_flag=False, *args, **kwargs):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        num_graphs = data.num_graphs
        # print('Input:',x.size(), edge_index.size(), edge_attr.size(), batch.size())

        hidden_node_feat = self.embedding_module(x)
        left_confidence = torch.ones([batch.size(0),1], dtype=x.dtype, device=x.device)
        # print('Embedded:',hidden_node_feat.size(), left_confidence.size())
        for iter_num in range(self.max_iter):
            if torch.max(left_confidence).item() > 1e-7:
                current_hidden_node_feat, current_confidence = self.body_module(
                    x = x, edge_index = edge_index, edge_attr=edge_attr, batch=batch,
                    hidden_node_feat=hidden_node_feat, left_confidence=left_confidence,
                    num_graphs=num_graphs,
                )
                # print('Body%d:'%iter_num,hidden_node_feat.size(), left_confidence.size()
                      # , current_confidence.size(), current_hidden_node_feat.size())
                hidden_node_feat = hidden_node_feat + current_confidence[batch]*current_hidden_node_feat
                left_confidence = 1.-current_confidence[batch]
                # print(hidden_node_feat.size(), left_confidence.size())
            else:
                break

        attention_x = torch.cat([x, hidden_node_feat], dim=-1)
        graph_feat = self.readout_module(
            x=hidden_node_feat, attention_x=attention_x
            , index=batch, size=num_graphs
        )

        out = self.head_module(graph_feat)

        output = out, scatter_add(x.new_ones(x.size(0)), batch, dim_size=num_graphs).reshape([-1,1])
        if output_node_feat:
            output = output + (hidden_node_feat, )
        if layer_num_flag:
            output = output + (iter_num, )
        return output

class NoSumSigmoidIterGNN(IterGNN):
    def forward(self, data, output_node_feat=False, layer_num_flag=False, *args, **kwargs):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        num_graphs = data.num_graphs
        # print('Input:',x.size(), edge_index.size(), edge_attr.size(), batch.size())

        hidden_node_feat = self.embedding_module(x)
        left_confidence = torch.ones([batch.size(0),1], dtype=x.dtype, device=x.device)
        # print('Embedded:',hidden_node_feat.size(), left_confidence.size())
        for iter_num in range(self.max_iter):
            if torch.max(left_confidence).item() > 1e-7:
                current_hidden_node_feat, current_confidence = self.body_module(
                    x = x, edge_index = edge_index, edge_attr=edge_attr, batch=batch,
                    hidden_node_feat=hidden_node_feat, left_confidence=left_confidence,
                    num_graphs=num_graphs,
                )
                # print('Body%d:'%iter_num,hidden_node_feat.size(), left_confidence.size()
                      # , current_confidence.size(), current_hidden_node_feat.size())
                hidden_node_feat = current_hidden_node_feat
                left_confidence = 1.-current_confidence[batch]
                # print(hidden_node_feat.size(), left_confidence.size())
            else:
                break
        hidden_node_feat = current_hidden_node_feat[batch]*hidden_node_feat

        attention_x = torch.cat([x, hidden_node_feat], dim=-1)
        graph_feat = self.readout_module(
            x=hidden_node_feat, attention_x=attention_x
            , index=batch, size=num_graphs
        )

        out = self.head_module(graph_feat)

        output = out, scatter_add(x.new_ones(x.size(0)), batch, dim_size=num_graphs).reshape([-1,1])
        if output_node_feat:
            output = output + (hidden_node_feat, )
        if layer_num_flag:
            output = output + (iter_num, )
        return output

class IterGNNV2(IterGNN):
    def forward(self, data, output_node_feat=False, layer_num_flag=False, *args, **kwargs):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        num_graphs = data.num_graphs
        # print('Input:',x.size(), edge_index.size(), edge_attr.size(), batch.size())

        hidden_node_feat = self.embedding_module(x)
        left_confidence = torch.ones([batch.size(0),1], dtype=x.dtype, device=x.device)
        # print('Embedded:',hidden_node_feat.size(), left_confidence.size())
        for iter_num in range(self.max_iter):
            if torch.max(left_confidence).item() > 1e-7:
                current_hidden_node_feat, current_confidence = self.body_module(
                    x = x, edge_index = edge_index, edge_attr=edge_attr, batch=batch,
                    hidden_node_feat=current_hidden_node_feat, left_confidence=left_confidence,
                    num_graphs=num_graphs,
                )
                # print('Body%d:'%iter_num,hidden_node_feat.size(), left_confidence.size()
                      # , current_confidence.size(), current_hidden_node_feat.size())
                hidden_node_feat = hidden_node_feat + left_confidence*current_confidence[batch]*current_hidden_node_feat
                left_confidence = left_confidence*(1.-current_confidence[batch])
                # print(hidden_node_feat.size(), left_confidence.size())
            else:
                break

        attention_x = torch.cat([x, hidden_node_feat], dim=-1)
        graph_feat = self.readout_module(
            x=hidden_node_feat, attention_x=attention_x
            , index=batch, size=num_graphs
        )

        out = self.head_module(graph_feat)

        output = out, scatter_add(x.new_ones(x.size(0)), batch, dim_size=num_graphs).reshape([-1,1])
        if output_node_feat:
            output = output + (hidden_node_feat, )
        if layer_num_flag:
            output = output + (iter_num, )
        return output

class AttMGNN(IterGNN):
    def forward(self, data, output_node_feat=False, layer_num_flag=False, *args, **kwargs):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        num_graphs = data.num_graphs
        # print('Input:',x.size(), edge_index.size(), edge_attr.size(), batch.size())

        hidden_node_feat = self.embedding_module(x)
        confidence_list, hidden_node_feat_list = [], []
        # print('Embedded:',hidden_node_feat.size(), left_confidence.size())
        left_confidence = torch.ones([batch.size(0),1], dtype=x.dtype, device=x.device)
        for iter_num in range(self.max_iter):
            current_hidden_node_feat, current_confidence = self.body_module(
                x = x, edge_index = edge_index, edge_attr=edge_attr, batch=batch,
                hidden_node_feat=hidden_node_feat, left_confidence=left_confidence,
                num_graphs=num_graphs,
            )
            hidden_node_feat = current_hidden_node_feat
            hidden_node_feat_list.append(current_hidden_node_feat)
            confidence_list.append(current_confidence)
        confidence = torch.stack(confidence_list, dim=-2)
        confidence = F.softmax(confidence, dim=-2)
        hidden_node_feat = torch.stack(hidden_node_feat_list, dim=-2)
        hidden_node_feat = torch.sum(confidence[batch] * hidden_node_feat, dim=-2).squeeze()

        attention_x = torch.cat([x, hidden_node_feat], dim=-1)
        graph_feat = self.readout_module(
            x=hidden_node_feat, attention_x=attention_x
            , index=batch, size=num_graphs
        )

        out = self.head_module(graph_feat)

        output = out, scatter_add(x.new_ones(x.size(0)), batch, dim_size=num_graphs).reshape([-1,1])
        if output_node_feat:
            output = output + (hidden_node_feat, )
        if layer_num_flag:
            output = output + (iter_num, )
        return output

class AttMGNNV2(IterGNN):
    def forward(self, data, output_node_feat=False, layer_num_flag=False, *args, **kwargs):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        num_graphs = data.num_graphs
        # print('Input:',x.size(), edge_index.size(), edge_attr.size(), batch.size())

        hidden_node_feat = self.embedding_module(x)
        confidence_list, hidden_node_feat_list = [], []
        # print('Embedded:',hidden_node_feat.size(), left_confidence.size())
        left_confidence = torch.ones([batch.size(0),1], dtype=x.dtype, device=x.device)
        for iter_num in range(self.max_iter):
            current_hidden_node_feat, current_confidence = self.body_module(
                x = x, edge_index = edge_index, edge_attr=edge_attr, batch=batch,
                hidden_node_feat=hidden_node_feat, left_confidence=left_confidence,
                num_graphs=num_graphs,
            )
            hidden_node_feat_list.append(current_hidden_node_feat)
            confidence_list.append(current_confidence)
            confidence = torch.stack(confidence_list, dim=-2)
            confidence = F.softmax(confidence, dim=-2)
            hidden_node_feat = torch.stack(hidden_node_feat_list, dim=-2)
            hidden_node_feat = torch.sum(confidence[batch] * hidden_node_feat, dim=-2).squeeze()

        attention_x = torch.cat([x, hidden_node_feat], dim=-1)
        graph_feat = self.readout_module(
            x=hidden_node_feat, attention_x=attention_x
            , index=batch, size=num_graphs
        )

        out = self.head_module(graph_feat)

        output = out, scatter_add(x.new_ones(x.size(0)), batch, dim_size=num_graphs).reshape([-1,1])
        if output_node_feat:
            output = output + (hidden_node_feat, )
        if layer_num_flag:
            output = output + (iter_num, )
        return output

class JKMGNN(IterGNN):
    def forward(self, data, output_node_feat=False, layer_num_flag=False, *args, **kwargs):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        num_graphs = data.num_graphs
        # print('Input:',x.size(), edge_index.size(), edge_attr.size(), batch.size())

        hidden_node_feat = self.embedding_module(x)
        confidence_list, hidden_node_feat_list = [], []
        # print('Embedded:',hidden_node_feat.size(), left_confidence.size())
        left_confidence = torch.ones([batch.size(0),1], dtype=x.dtype, device=x.device)
        for iter_num in range(self.max_iter):
            current_hidden_node_feat, current_confidence = self.body_module(
                x = x, edge_index = edge_index, edge_attr=edge_attr, batch=batch,
                hidden_node_feat=hidden_node_feat, left_confidence=left_confidence,
                num_graphs=num_graphs,
            )
            hidden_node_feat = current_hidden_node_feat
            hidden_node_feat_list.append(current_hidden_node_feat)
            confidence_list.append(current_confidence)
        hidden_node_feat = torch.stack(hidden_node_feat_list, dim=-2)
        hidden_node_feat = torch.max(hidden_node_feat, dim=-2)[0].squeeze()

        attention_x = torch.cat([x, hidden_node_feat], dim=-1)
        graph_feat = self.readout_module(
            x=hidden_node_feat, attention_x=attention_x
            , index=batch, size=num_graphs
        )

        out = self.head_module(graph_feat)

        output = out, scatter_add(x.new_ones(x.size(0)), batch, dim_size=num_graphs).reshape([-1,1])
        if output_node_feat:
            output = output + (hidden_node_feat, )
        if layer_num_flag:
            output = output + (iter_num, )
        return output

class JKMGNNV2(IterGNN):
    def forward(self, data, output_node_feat=False, layer_num_flag=False, *args, **kwargs):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        num_graphs = data.num_graphs
        # print('Input:',x.size(), edge_index.size(), edge_attr.size(), batch.size())

        hidden_node_feat = self.embedding_module(x)
        confidence_list, hidden_node_feat_list = [], []
        # print('Embedded:',hidden_node_feat.size(), left_confidence.size())
        left_confidence = torch.ones([batch.size(0),1], dtype=x.dtype, device=x.device)
        for iter_num in range(self.max_iter):
            current_hidden_node_feat, current_confidence = self.body_module(
                x = x, edge_index = edge_index, edge_attr=edge_attr, batch=batch,
                hidden_node_feat=hidden_node_feat, left_confidence=left_confidence,
                num_graphs=num_graphs,
            )
            hidden_node_feat = current_hidden_node_feat
            hidden_node_feat_list.append(current_hidden_node_feat)
            confidence_list.append(current_confidence)
            hidden_node_feat = torch.stack(hidden_node_feat_list, dim=-2)
            hidden_node_feat = torch.max(hidden_node_feat, dim=-2)[0].squeeze()

        attention_x = torch.cat([x, hidden_node_feat], dim=-1)
        graph_feat = self.readout_module(
            x=hidden_node_feat, attention_x=attention_x
            , index=batch, size=num_graphs
        )

        out = self.head_module(graph_feat)

        output = out, scatter_add(x.new_ones(x.size(0)), batch, dim_size=num_graphs).reshape([-1,1])
        if output_node_feat:
            output = output + (hidden_node_feat, )
        if layer_num_flag:
            output = output + (iter_num, )
        return output

class JKCGNN(JKMGNN):
    def __init__(self, hidden_size=64, *args, **kwargs):
        super(JKCGNN, self).__init__(*args, **kwargs)
        self.node_feat_embed_module = MLP([self.max_iter*hidden_size, hidden_size])
    def forward(self, data, output_node_feat=False, layer_num_flag=False, *args, **kwargs):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        num_graphs = data.num_graphs
        # print('Input:',x.size(), edge_index.size(), edge_attr.size(), batch.size())

        hidden_node_feat = self.embedding_module(x)
        confidence_list, hidden_node_feat_list = [], []
        # print('Embedded:',hidden_node_feat.size(), left_confidence.size())
        left_confidence = torch.ones([batch.size(0),1], dtype=x.dtype, device=x.device)
        for iter_num in range(self.max_iter):
            current_hidden_node_feat, current_confidence = self.body_module(
                x = x, edge_index = edge_index, edge_attr=edge_attr, batch=batch,
                hidden_node_feat=hidden_node_feat, left_confidence=left_confidence,
                num_graphs=num_graphs,
            )
            hidden_node_feat = current_hidden_node_feat
            hidden_node_feat_list.append(current_hidden_node_feat)
            confidence_list.append(current_confidence)
        hidden_node_feat = torch.cat(hidden_node_feat_list, dim=-1)
        hidden_node_feat = self.node_feat_embed_module(hidden_node_feat)

        attention_x = torch.cat([x, hidden_node_feat], dim=-1)
        graph_feat = self.readout_module(
            x=hidden_node_feat, attention_x=attention_x
            , index=batch, size=num_graphs
        )

        out = self.head_module(graph_feat)

        output = out, scatter_add(x.new_ones(x.size(0)), batch, dim_size=num_graphs).reshape([-1,1])
        if output_node_feat:
            output = output + (hidden_node_feat, )
        if layer_num_flag:
            output = output + (iter_num, )
        return output

class Body(nn.Module):
    def __init__(self, edge_embedding_module, aggregation_module, update_module,
                 readout_module, score_module, *args, **kwargs):
        super(Body, self).__init__()
        self.edge_embedding_module = edge_embedding_module
        self.aggregation_module = aggregation_module
        self.update_module = update_module
        self.readout_module = readout_module
        self.score_module = score_module
    def forward(self, *args, **kwargs):
        x = kwargs.get('x')
        node_feat = kwargs.get('hidden_node_feat')
        edge_index = kwargs.get('edge_index')
        edge_attr = kwargs.get('edge_attr')
        batch = kwargs.get('batch')
        num_nodes = x.size(0)
        num_graphs = kwargs.get('num_graphs')
        # print('Body/Input:',x.size(), node_feat.size(), edge_index.size(), edge_attr.size())

        rol, col = edge_index
        edge_feat = torch.cat([x[rol], node_feat[rol], edge_attr], dim=-1)
        edge_att_feat = torch.cat([x[rol], node_feat[rol]
                                   , x[col], node_feat[col]
                                   , edge_attr], dim=-1)
        edge_feat_embedded = self.edge_embedding_module(edge_feat)
        node_feat_cand = self.aggregation_module(x=edge_feat_embedded,
                                             attention_x=edge_att_feat,
                                             index=col,
                                             size=num_nodes)

        # print('Body/Edge_embedded:',node_feat.size(), node_feat_cand.size(), x.size())
        update_att_feat = torch.cat([node_feat, node_feat_cand, x], dim=-1)
        node_feat = self.update_module(x=node_feat,
                                       cand_x = node_feat_cand,
                                       attention_x = update_att_feat)
        node_feat_att = torch.cat([node_feat, x], dim=-1)
        graph_feat = self.readout_module(x=node_feat,
                                         attention_x=node_feat_att,
                                         index=batch,
                                         size=num_graphs,)
        confidence = torch.sigmoid(self.score_module(graph_feat))
        return node_feat, confidence

class _Update(nn.Module):
    def __init__(self, *args, **kwargs):
        super(_Update,self).__init__()
    def forward(self,*args, **kwargs):
        raise NotImplementedError

class IdentityUpdate(_Update):
    def forward(self,x, cand_x, *args, **kwargs):
        return cand_x

class MaxUpdate(_Update):
    def forward(self,x, cand_x, *args, **kwargs):
        return torch.max(x, cand_x)

class MeanUpdate(_Update):
    def forward(self, x, cand_x, *args, **kwargs):
        return torch.mean(x, cand_x)

class AddUpdate(_Update):
    def forward(self, x, cand_x, *args, **kwargs):
        return torch.add(x, cand_x)

class GateUpdate(nn.Module):
    def __init__(self, gate_module, update_module):
        super(GateUpdate, self).__init__()
        self.gate_module =  gate_module
        self.update_module = update_module
    def forward(self, x, cand_x, attention_x, *args, **kwargs):
        gate = torch.sigmoid(self.gate_module(attention_x))
        x = x * gate + cand_x
        return self.update_module(x)

class BodyV2(Body):
    'simplified version'
    def forward(self, *args, **kwargs):
        x = kwargs.get('x')
        node_feat = kwargs.get('hidden_node_feat')
        edge_index = kwargs.get('edge_index')
        edge_attr = kwargs.get('edge_attr')
        batch = kwargs.get('batch')
        num_nodes = x.size(0)
        num_graphs = kwargs.get('num_graphs')
        # print('Body/Input:',x.size(), node_feat.size(), edge_index.size(), edge_attr.size())

        rol, col = edge_index
        edge_feat = torch.cat([node_feat[rol], edge_attr], dim=-1)
        edge_att_feat = torch.cat([node_feat[rol]
                                   , node_feat[col]
                                   , edge_attr], dim=-1)
        edge_feat_embedded = self.edge_embedding_module(edge_feat)
        node_feat_cand = self.aggregation_module(x=edge_feat_embedded,
                                             attention_x=edge_att_feat,
                                             index=col,
                                             size=num_nodes)

        # print('Body/Edge_embedded:',node_feat.size(), node_feat_cand.size(), x.size())
        update_att_feat = torch.cat([node_feat, node_feat_cand], dim=-1)
        node_feat = self.update_module(x=node_feat,
                                       cand_x = node_feat_cand,
                                       attention_x = update_att_feat)
        return node_feat

class GenBody(Body):
    def forward(self, *args, **kwargs):
        x = kwargs.get('x')
        node_feat = kwargs.get('hidden_node_feat')
        edge_index = kwargs.get('edge_index')
        edge_attr = kwargs.get('edge_attr')
        batch = kwargs.get('batch')
        num_nodes = x.size(0)
        num_graphs = kwargs.get('num_graphs')
        # print('Body/Input:',x.size(), node_feat.size(), edge_index.size(), edge_attr.size())

        rol, col = edge_index
        edge_feat = torch.cat([x[rol], node_feat[rol],
                               x[col], node_feat[col],
                               edge_attr], dim=-1)
        edge_att_feat = torch.cat([x[rol], node_feat[rol]
                                   , x[col], node_feat[col]
                                   , edge_attr], dim=-1)
        edge_feat_embedded = self.edge_embedding_module(edge_feat)
        node_feat_cand = self.aggregation_module(x=edge_feat_embedded,
                                             attention_x=edge_att_feat,
                                             index=col,
                                             size=num_nodes)

        # print('Body/Edge_embedded:',node_feat.size(), node_feat_cand.size(), x.size())
        update_att_feat = torch.cat([node_feat, node_feat_cand, x], dim=-1)
        node_feat = self.update_module(x=node_feat,
                                       cand_x = node_feat_cand,
                                       attention_x = update_att_feat)
        node_feat_att = torch.cat([node_feat, x], dim=-1)
        graph_feat = self.readout_module(x=node_feat,
                                         attention_x=node_feat_att,
                                         index=batch,
                                         size=num_graphs,)
        confidence = torch.sigmoid(self.score_module(graph_feat))
        return node_feat, confidence

class GNNBody(Body):
    def __init__(self, gnn_module=None, *args, **kwargs):
        super(GNNBody, self).__init__(*args, **kwargs)
        self.gnn_module = gnn_module
    def forward(self, *args, **kwargs):
        x = kwargs.get('x')
        node_feat = kwargs.get('hidden_node_feat')
        edge_index = kwargs.get('edge_index')
        batch = kwargs.get('batch')
        num_nodes = x.size(0)
        num_graphs = kwargs.get('num_graphs')

        node_feat = self.gnn_module(node_feat, edge_index)
        node_feat_att = torch.cat([node_feat, x], dim=-1)
        graph_feat = self.readout_module(x=node_feat,
                                         attention_x=node_feat_att,
                                         index=batch,
                                         size=num_graphs,)
        confidence = torch.sigmoid(self.score_module(graph_feat))
        return node_feat, confidence

def Baseline(in_channels, edge_channels, out_channels,
             net_name='IterGNN', max_iter=30,
             hidden_size=64, embedding_layer_num=2,
             edge_embedding_layer_num=2,
             aggregation_name='Max', aggregation_score_layer_num=1,
             update_module_name='Max', update_gate_layer_num=1,
             readout_name='Attention', readout_score_layer_num=1,
             confidence_layer_num=1,
             head_layer_num=1,
             last_bias=True,
             bias_flag=True,
             *args, **kwargs,
             ):

    embedding_size_list = cal_size_list(in_channels, hidden_size, embedding_layer_num)
    embedding_module = MLP(embedding_size_list, last_bias=last_bias, bias_flag=bias_flag)

    edge_embedding_size_list = cal_size_list(
        in_channels+edge_channels+(in_channels if embedding_layer_num == 0 else hidden_size)
        , hidden_size, edge_embedding_layer_num
    )
    edge_embedding_module = MLP(edge_embedding_size_list, last_bias=last_bias, bias_flag=bias_flag)

    aggregation_score_size_list = cal_size_list(
        edge_channels+in_channels*2+2*(in_channels if embedding_layer_num == 0 else hidden_size),
        1, aggregation_score_layer_num,
    )
    aggregation_score_module = MLP(aggregation_score_size_list, nn.Identity, last_bias=False, bias_flag=bias_flag)
    aggregation_embedding_module = nn.Identity()
    aggregation_module = globals().get(aggregation_name+'Aggregation')(
        score_module = aggregation_score_module,
        embedding_module = aggregation_embedding_module
    )

    update_gate_size_list = cal_size_list(in_channels+2*hidden_size, 1, update_gate_layer_num)
    update_gate_module =  MLP(update_gate_size_list, nn.Identity, bias_flag=bias_flag)
    update_update_module = nn.Identity()
    update_module = globals().get(update_module_name+'Update')(
        gate_module=update_gate_module,
        update_module=update_update_module
    )

    readout_score_size_list = cal_size_list(in_channels+hidden_size, 1, readout_score_layer_num)
    readout_score_module = MLP(readout_score_size_list, nn.Identity, last_bias=False, bias_flag=bias_flag)
    readout_embedding_module = nn.Identity()
    readout_module = globals().get(readout_name+'Aggregation')(
        score_module = readout_score_module,
        embedding_module = readout_embedding_module
    )

    score_size_list= cal_size_list(hidden_size, 1, confidence_layer_num)
    score_module = MLP(score_size_list, nn.Identity, bias_flag=bias_flag)

    body_module = Body(
        edge_embedding_module=edge_embedding_module,
        aggregation_module=aggregation_module,
        update_module=update_module,
        readout_module=readout_module,
        score_module=score_module
    )

    head_size_list = cal_size_list(hidden_size, out_channels, head_layer_num)
    head_module = MLP(head_size_list, nn.Identity, bias_flag=bias_flag)

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

def GenBaseline(in_channels, edge_channels, out_channels,
             net_name='IterGNN', max_iter=30,
             body_name='GenBody', body_conv_name='GAT',
             hidden_size=64, embedding_layer_num=2,
             edge_embedding_layer_num=2,
             aggregation_name='Max', aggregation_score_layer_num=1, aggregation_size_layer_num=2,
             update_module_name='Max', update_gate_layer_num=1,
             readout_name='Attention', readout_score_layer_num=1, readout_size_layer_num=2,
             confidence_layer_num=1,
             head_layer_num=1,
             last_bias=True,
             bias_flag=True,
             *args, **kwargs,
             ):

    embedding_size_list = cal_size_list(in_channels, hidden_size, embedding_layer_num)
    embedding_module = MLP(embedding_size_list, last_bias=last_bias, bias_flag=bias_flag)

    edge_embedding_size_list = cal_size_list(
        2*in_channels+edge_channels+2*(in_channels if embedding_layer_num == 0 else hidden_size)
        , hidden_size, edge_embedding_layer_num
    )
    edge_embedding_module = MLP(edge_embedding_size_list, last_bias=last_bias, bias_flag=bias_flag)

    aggregation_score_size_list = cal_size_list(
        edge_channels+in_channels*2+2*(in_channels if embedding_layer_num == 0 else hidden_size),
        1, aggregation_score_layer_num,
    )
    aggregation_score_module = MLP(aggregation_score_size_list, nn.Identity, last_bias=False, bias_flag=bias_flag)
    aggregation_embedding_module = nn.Identity()
    aggregation_size_size_list = [1]+[hidden_size]*(aggregation_size_layer_num-1)+[1]
    aggregation_size_module = MLP(aggregation_size_size_list, last_bias=last_bias, bias_flag=bias_flag)
    aggregation_module = globals().get(aggregation_name+'Aggregation')(
        score_module = aggregation_score_module,
        embedding_module = aggregation_embedding_module,
        size_update_module=aggregation_size_module,
    )

    update_gate_size_list = cal_size_list(in_channels+2*hidden_size, 1, update_gate_layer_num)
    update_gate_module =  MLP(update_gate_size_list, nn.Identity, bias_flag=bias_flag)
    update_update_module = nn.Identity()
    update_module = globals().get(update_module_name+'Update')(
        gate_module=update_gate_module,
        update_module=update_update_module
    )

    readout_score_size_list = cal_size_list(in_channels+hidden_size, 1, readout_score_layer_num)
    readout_score_module = MLP(readout_score_size_list, nn.Identity, last_bias=False, bias_flag=bias_flag)
    readout_embedding_module = nn.Identity()
    readout_size_size_list = [1]+[hidden_size]*(readout_size_layer_num-1)+[1]
    readout_size_module = MLP(readout_size_size_list, last_bias=last_bias, bias_flag=bias_flag)
    readout_module = globals().get(readout_name+'Aggregation')(
        score_module = readout_score_module,
        embedding_module = readout_embedding_module,
        size_update_module=readout_size_module,
    )

    score_size_list = cal_size_list(hidden_size, 1, confidence_layer_num)
    score_module = MLP(score_size_list, nn.Identity, bias_flag=bias_flag)

    body_gnn_module = MySequential(
        globals()[body_conv_name](hidden_size, hidden_size, bias=bias_flag),
        nn.LeakyReLU(),
    )
    body_module = globals()[body_name](
        edge_embedding_module=edge_embedding_module,
        aggregation_module=aggregation_module,
        update_module=update_module,
        readout_module=readout_module,
        score_module=score_module,
        gnn_module=body_gnn_module,
    )

    head_size_list = cal_size_list(hidden_size, out_channels, head_layer_num)
    head_module = MLP(head_size_list, nn.Identity, bias_flag=bias_flag)

    model_type = globals().get(net_name)
    model = model_type(
        embedding_module = embedding_module,
        body_module = body_module,
        readout_module = readout_module,
        head_module = head_module,
        max_iter = max_iter,
        hidden_size=hidden_size,
    )
    print('here: ', model)

    return model

class MultiRes(nn.Module):
    def __init__(self, block_module_func, in_channels, out_channels, hidden_size=64,
                 res_layer_num=3, max_iter=30, *args, **kwargs):
        super(MultiRes, self).__init__()
        layer_num_index = lambda i:int(i*max_iter/res_layer_num)
        layer_num = lambda i: layer_num_index(i+1) - layer_num_index(i)
        self.blocks = nn.ModuleList([
            block_module_func(
                in_channels=(in_channels if i==0 else hidden_size),
                hidden_size = hidden_size,
                out_channels=out_channels,
                max_iter=layer_num(i), *args, **kwargs
            )
            for i in range(res_layer_num)
        ])
    def forward(self, data):
        _, _, feat = self.blocks[0](data, output_node_feat=True)
        data.x = feat
        for b in self.blocks[1:-1]:
            _, _, feat = b(data, output_node_feat=True)
            data.x = data.x + feat
        out = self.blocks[-1](data)
        return out

def ResBaseline(*args, **kwargs):
    return MultiRes(Baseline, *args ,**kwargs)

def ResGenBaseline(*args, **kwargs):
    return MultiRes(GenBaseline, *args, **kwargs)

class MultiCat(nn.Module):
    def __init__(self, block_module_func, in_channels, out_channels, hidden_size=64,
                 res_layer_num=3, max_iter=30, *args, **kwargs):
        super(MultiCat, self).__init__()
        layer_num_index = lambda i:int(i*max_iter/res_layer_num)
        layer_num = lambda i: layer_num_index(i+1) - layer_num_index(i)
        self.blocks = nn.ModuleList([
            block_module_func(
                in_channels=(in_channels if i==0 else hidden_size),
                hidden_size = hidden_size,
                out_channels= hidden_size,
                max_iter=layer_num(i), *args, **kwargs
            )
            for i in range(res_layer_num)
        ])
    def forward(self, data):
        _, _, feat = self.blocks[0](data, output_node_feat=True)
        data.x = feat
        out_list, layer_num_list = [], []
        for b in self.blocks[1:-1]:
            out, layer_num, feat = b(data, output_node_feat=True)
            data.x = feat
            out_list.append(out)
            layer_num_list.append(layer_num)
        out, layer_num = self.blocks[-1](data)
        out_list.append(out)
        layer_num_list.append(layer_num)
        out = torch.sum(torch.stack(out_list, dim=0), dim=0)
        return out, torch.max(torch.stack(layer_num_list, dim=0), dim=0)

def CatBaseline(*args, **kwargs):
    return MultiCat(Baseline, *args ,**kwargs)

def CatGenBaseline(*args, **kwargs):
    return MultiCat(GenBaseline, *args, **kwargs)

