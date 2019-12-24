import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import BatchNorm1d, LayerNorm

from torch_geometric.data import Batch, Data
from torch_geometric.nn import EdgeConv, GCNConv, GATConv
from torch_geometric.utils import scatter_

# from .utils import MyGCNConv as GCNConv
from .utils import MyEdgeAttConv, ZeroLinear
from .utils import cal_size_list, MLP
from .utils import MySequential
from .utils import get_activation_func
from .aggregation import get_readout_func
from .utils import MyGINConv as GINConv, MyGINConvV2 as GINConvV2
from .utils import MyMeanGINConv as MeanGINConv, MyMeanGINConvV2 as MeanGINConvV2


class GNNNet(nn.Module):
    def __init__(self, in_channels, out_channels
                 , hidden_size=64, embedding_layer_num=1
                 , conv_layer_num=2, conv_name='GATConv'
                 , readout_name='first'
                 , activation_name='LeakyReLU'
                 , head_layer_num=1
                 , **kwargs):
        super(GNNNet, self).__init__()

        # parameters
        conv_type = globals()[conv_name]
        self.norm_flag = True
        self.readout_func = get_readout_func(readout_name)
        self.activation_type = get_activation_func(activation_name)

        # attributes
        self.embeddings = None
        self.convs = None
        self.heads = None

        # Embedding Layers
        if embedding_layer_num > 0:
            embedding_dim_list = np.linspace(
                in_channels, hidden_size
                , embedding_layer_num+1, dtype='int'
            )
            self.embeddings = nn.ModuleList(
                [nn.Sequential(
                    *(
                        (nn.Linear(embedding_dim_list[ln], embedding_dim_list[ln+1], bias=True), )
                        + (self.activation_type(), )
                    )
                ) for ln in range(embedding_layer_num)]
            )

        # Convolution Layers
        if conv_layer_num > 0:
            conv_dim_list = np.linspace(
                start = in_channels if self.embeddings is None else hidden_size,
                stop= hidden_size, num=conv_layer_num+1, dtype='int'
            )
            self.convs = nn.ModuleList(
                [MySequential(
                    *(
                        (conv_type(conv_dim_list[ln], conv_dim_list[ln+1], bias=True), )
                        + (self.activation_type(), )
                    )
                ) for ln in range(conv_layer_num)]
            )

        # Head Layers
        assert head_layer_num > 0
        head_dim_list = np.linspace(
            in_channels if self.embeddings is None and self.convs is None else hidden_size,
            out_channels, head_layer_num+1, dtype='int'
        )
        self.heads = nn.ModuleList(
            [nn.Sequential(
                *(
                    (nn.Linear(head_dim_list[ln], head_dim_list[ln+1], bias=True), )
                    + ((self.activation_type(), ) if ln != head_layer_num-1 else tuple())
                )
            ) for ln in range(head_layer_num)]
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        num_nodes = scatter_('add',x.new_ones(x.size(0)), batch, dim_size=data.num_graphs)

        # embedding
        if self.embeddings is not None:
            for layer in self.embeddings:
                x = layer(x)

        # conv
        if self.convs is not None:
            for layer in self.convs:
                x = layer(x, edge_index)

        # pool
        x = self.readout_func(x, batch, size=data.num_graphs)

        # heads
        for layer in self.heads:
            x = layer(x)

        return x, num_nodes.reshape([-1,1])

class GEATNet(nn.Module):
    def __init__(self, in_channels, out_channels
                 , hidden_size=64, embedding_layer_num=1
                 , aggregation_score_layer_num=2, aggregation_score_zero_init=False
                 , readout_name='first'
                 , activation_name='LeakyReLU'
                 , head_layer_num=1
                 , **kwargs):
        super(GEATNet, self).__init__()

        self.readout_func = get_readout_func(readout_name)
        self.activation_type = get_activation_func(activation_name)

        # Embedding Layers
        embedding_dim_list = cal_size_list(in_channels, hidden_size, embedding_layer_num)
        embeddings = MLP(embedding_dim_list, activation=self.activation_type, last_activation=self.activation_type)

        # Attention Layers
        attentions = None
        attention_dim_list = [in_channels,]+[hidden_size]*(aggregation_score_layer_num-1)+[1,]
        attentions = nn.Sequential(
            *(nn.Sequential(
                *(
                    ((nn.Linear if not aggregation_score_zero_init else ZeroLinear)(attention_dim_list[ln], attention_dim_list[ln+1], bias=True), )
                    + ((self.activation_type(), ) if ln < aggregation_score_layer_num-1 else tuple())
                )
            ) for ln in range(aggregation_score_layer_num))
        )
        # Update Layers
        updates = nn.Identity()

        self.edge_att_conv = MyEdgeAttConv(embeddings, attentions, updates)

        # Head Layers
        assert head_layer_num > 0
        head_dim_list = cal_size_list(hidden_size, out_channels, head_layer_num)
        self.heads = MLP(head_dim_list, activation=self.activation_type, last_activation=nn.Identity)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        num_nodes = scatter_('add',x.new_ones(x.size(0)), batch, dim_size=data.num_graphs)

        x = self.edge_att_conv(x, edge_index)
        x = self.readout_func(x, batch, size=data.num_graphs)
        for layer in self.heads:
            x = layer(x)

        return x, num_nodes.reshape([-1,1])
