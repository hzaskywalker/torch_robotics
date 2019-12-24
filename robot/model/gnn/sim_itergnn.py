#!/usr/bin/env python
# coding=utf-8

from .itergnn import *

class IterGNNV2(IterGNN):
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        num_graphs = data.num_graphs

        hidden_node_feat = [self.embedding_module(x),]
        confidence = [torch.zeros([num_graphs,1], dtype=x.dtype, device=x.device),]
        for iter_num in range(self.max_iter):
            current_hidden_node_feat, current_confidence = self.body_module(
                x = x, edge_index = edge_index, edge_attr=edge_attr, batch=batch,
                hidden_node_feat=hidden_node_feat[-1], confidence=confidence[-1],
                num_graphs=num_graphs,
            )
            hidden_node_feat.append(current_hidden_node_feat)
            confidence.append(current_confidence)
        confidence = torch.stack(confidence, dim=0)[:,batch]
        hidden_node_feat = torch.stack(hidden_node_feat, dim=0)
        weights = F.softmax(confidence*10, dim=0)
        hidden_node_feat = torch.sum(weights * hidden_node_feat, dim=0)

        attention_x = torch.cat([x, hidden_node_feat], dim=-1)
        graph_feat = self.readout_module(
            x=hidden_node_feat, attention_x=attention_x
            , index=batch, size=num_graphs
        )

        out = self.head_module(graph_feat)

        return out, scatter_add(x.new_ones(x.size(0)), batch, dim_size=num_graphs).reshape([-1,1])

class IterGNNV3(IterGNN):
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        num_graphs = data.num_graphs
        graph_sizes = scatter_add(torch.ones([x.size(0),],dtype=x.dtype,device=x.device), batch, dim_size=num_graphs)
        max_graph_sizes = torch.max(graph_sizes)

        hidden_node_feat = self.embedding_module(x)
        # confidence = [torch.zeros([num_graphs,1], dtype=x.dtype, device=x.device),]
        confidence = torch.zeros([num_graphs, 1], dtype=x.dtype, device=x.device)
        for iter_num in range(int(max_graph_sizes.item()-1)):
            hidden_node_feat = self.body_module(
                x = x, edge_index = edge_index, edge_attr=edge_attr, batch=batch,
                hidden_node_feat=hidden_node_feat, confidence=confidence,
                num_graphs=num_graphs,
            )

        attention_x = torch.cat([x, hidden_node_feat], dim=-1)
        graph_feat = self.readout_module(
            x=hidden_node_feat, attention_x=attention_x
            , index=batch, size=num_graphs
        )

        out = self.head_module(graph_feat)

        return out, scatter_add(x.new_ones(x.size(0)), batch, dim_size=num_graphs).reshape([-1,1])

class IterGNNV4(IterGNN):
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        num_graphs = data.num_graphs
        graph_sizes = scatter_add(torch.ones([x.size(0),],dtype=x.dtype,device=x.device), batch, dim_size=num_graphs)
        max_graph_sizes = torch.max(graph_sizes)

        hidden_node_feat = self.embedding_module(x)
        # confidence = [torch.zeros([num_graphs,1], dtype=x.dtype, device=x.device),]
        confidence = torch.zeros([num_graphs, 1], dtype=x.dtype, device=x.device)
        for iter_num in range(int(max_graph_sizes.item()-1)):
            hidden_node_feat = self.body_module(
                x = x, edge_index = edge_index, edge_attr=edge_attr, batch=batch,
                hidden_node_feat=hidden_node_feat, confidence=confidence,
                num_graphs=num_graphs,
            )

        attention_x = torch.cat([x], dim=-1)
        graph_feat = self.readout_module(
            x=hidden_node_feat, attention_x=attention_x
            , index=batch, size=num_graphs
        )

        out = self.head_module(graph_feat)

        return out, scatter_add(x.new_ones(x.size(0)), batch, dim_size=num_graphs).reshape([-1,1])

def SimpleBaseline(in_channels, edge_channels, out_channels,
             net_name='IterGNNV3',
             hidden_size=4, embedding_layer_num=1,
             edge_embedding_layer_num=1,
             aggregation_name='Max', aggregation_score_layer_num=1,
             update_module_name='Identity', update_gate_layer_num=1,
             readout_name='Attention', readout_score_layer_num=1,
             confidence_layer_num=1,
             head_layer_num=1,
             *args, **kwargs,
             ):

    embedding_size_list = cal_size_list(in_channels, hidden_size, embedding_layer_num)
    embedding_module = MLP(embedding_size_list,)

    edge_embedding_size_list = cal_size_list(
        edge_channels+hidden_size
        , hidden_size, edge_embedding_layer_num
    )
    edge_embedding_module = MLP(edge_embedding_size_list)

    aggregation_score_size_list = cal_size_list(
        edge_channels+2*hidden_size,
        1, aggregation_score_layer_num,
    )
    aggregation_score_module = MLP(aggregation_score_size_list, nn.Identity)
    aggregation_embedding_module = nn.Identity()
    aggregation_module = globals().get(aggregation_name+'Aggregation')(
        score_module = aggregation_score_module,
        embedding_module = aggregation_embedding_module
    )

    update_gate_size_list = cal_size_list(2*hidden_size, 1, update_gate_layer_num)
    update_gate_module =  MLP(update_gate_size_list, nn.Identity)
    update_update_module = nn.Identity()
    update_module = globals().get(update_module_name+'Update')(
        gate_module=update_gate_module,
        update_module=update_update_module
    )

    readout_score_size_list = cal_size_list(in_channels+hidden_size, 1, readout_score_layer_num)
    readout_score_module = MLP(readout_score_size_list, nn.Identity)
    readout_embedding_module = nn.Identity()
    readout_module = globals().get(readout_name+'Aggregation')(
        score_module = readout_score_module,
        embedding_module = readout_embedding_module
    )

    score_size_list= cal_size_list(hidden_size, 1, confidence_layer_num)
    score_module = MLP(score_size_list, nn.Identity)

    body_module = BodyV2(
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
        max_iter = None,
    )

    return model

def SimpleBaselineV2(in_channels, edge_channels, out_channels,
             net_name='IterGNNV4',
             hidden_size=4, embedding_layer_num=1,
             edge_embedding_layer_num=1,
             aggregation_name='Max', aggregation_score_layer_num=1,
             update_module_name='Identity', update_gate_layer_num=1,
             readout_name='Attention', readout_score_layer_num=1,
             confidence_layer_num=1,
             head_layer_num=1,
             *args, **kwargs,
             ):

    embedding_size_list = cal_size_list(in_channels, hidden_size, embedding_layer_num)
    embedding_module = MLP(embedding_size_list, nn.Identity)

    edge_embedding_size_list = cal_size_list(
        edge_channels+hidden_size
        , hidden_size, edge_embedding_layer_num
    )
    edge_embedding_module = MLP(edge_embedding_size_list, nn.Identity)

    aggregation_score_size_list = cal_size_list(
        edge_channels+2*hidden_size,
        1, aggregation_score_layer_num,
    )
    aggregation_score_module = MLP(aggregation_score_size_list, nn.Identity)
    aggregation_embedding_module = nn.Identity()
    aggregation_module = globals().get(aggregation_name+'Aggregation')(
        score_module = aggregation_score_module,
        embedding_module = aggregation_embedding_module
    )

    update_gate_size_list = cal_size_list(2*hidden_size, 1, update_gate_layer_num)
    update_gate_module =  MLP(update_gate_size_list, nn.Identity)
    update_update_module = nn.Identity()
    update_module = globals().get(update_module_name+'Update')(
        gate_module=update_gate_module,
        update_module=update_update_module
    )

    readout_score_size_list = cal_size_list(in_channels, 1, readout_score_layer_num)
    readout_score_module = MLP(readout_score_size_list, nn.Identity)
    readout_embedding_module = nn.Identity()
    readout_module = globals().get(readout_name+'Aggregation')(
        score_module = readout_score_module,
        embedding_module = readout_embedding_module
    )

    score_size_list = cal_size_list(hidden_size, 1, confidence_layer_num)
    score_module = MLP(score_size_list, nn.Identity)

    body_module = BodyV2(
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
        max_iter = None,
    )

    return model

def SimpleBaselineV3(in_channels, edge_channels, out_channels,
             net_name='IterGNNV4',
             hidden_size=4, embedding_layer_num=1,
             edge_embedding_layer_num=1,
             aggregation_name='Max', aggregation_score_layer_num=1,
             update_module_name='Identity', update_gate_layer_num=1,
             readout_name='Attention', readout_score_layer_num=1,
             confidence_layer_num=1,
             head_layer_num=1,
             *args, **kwargs,
             ):

    embedding_size_list = cal_size_list(in_channels, hidden_size, embedding_layer_num)
    embedding_module = MLP(embedding_size_list, nn.Identity)

    edge_embedding_size_list = cal_size_list(
        edge_channels+hidden_size
        , hidden_size, edge_embedding_layer_num
    )
    edge_embedding_module = MLP(edge_embedding_size_list, nn.Identity)

    aggregation_score_size_list = cal_size_list(
        edge_channels+2*hidden_size,
        1, aggregation_score_layer_num,
    )
    aggregation_score_module = MLP(aggregation_score_size_list, nn.Identity)
    aggregation_embedding_module = nn.Identity()
    aggregation_module = globals().get(aggregation_name+'Aggregation')(
        score_module = aggregation_score_module,
        embedding_module = aggregation_embedding_module
    )

    update_gate_size_list = cal_size_list(2*hidden_size, 1, update_gate_layer_num)
    update_gate_module =  MLP(update_gate_size_list, nn.Identity)
    update_update_module = nn.Identity()
    update_module = globals().get(update_module_name+'Update')(
        gate_module=update_gate_module,
        update_module=update_update_module
    )

    # readout_score_size_list = cal_size_list(in_channels, 1, readout_score_layer_num)
    # readout_score_module = MLP(readout_score_size_list, nn.Identity)
    # readout_embedding_module = nn.Identity()
    # readout_module = globals().get(readout_name+'Aggregation')(
        # score_module = readout_score_module,
        # embedding_module = readout_embedding_module
    # )
    readout_module = lambda x, attention_x, *args, **kwargs: x[attention_x[:,1]==1]

    score_size_list = cal_size_list(hidden_size, 1, confidence_layer_num)
    score_module = MLP(score_size_list, nn.Identity)

    body_module = BodyV2(
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
        max_iter = None,
    )

    return model

def SimpleBaselineV4(in_channels, edge_channels, out_channels,
             net_name='IterGNNV4',
             hidden_size=4, embedding_layer_num=1,
             edge_embedding_layer_num=1,
             aggregation_name='Max', aggregation_score_layer_num=1,
             update_module_name='Identity', update_gate_layer_num=1,
             readout_name='Attention', readout_score_layer_num=1,
             confidence_layer_num=1,
             head_layer_num=1,
             *args, **kwargs,
             ):

    embedding_size_list = cal_size_list(in_channels, hidden_size, embedding_layer_num)
    embedding_module = MLP(embedding_size_list, nn.Identity)

    edge_embedding_size_list = cal_size_list(
        edge_channels+hidden_size
        , hidden_size, edge_embedding_layer_num
    )
    edge_embedding_module = MLP(edge_embedding_size_list, nn.Identity)

    aggregation_score_size_list = cal_size_list(
        edge_channels+2*hidden_size,
        1, aggregation_score_layer_num,
    )
    aggregation_score_module = MLP(aggregation_score_size_list, nn.Identity)
    aggregation_embedding_module = nn.Identity()
    aggregation_module = globals().get(aggregation_name+'Aggregation')(
        score_module = aggregation_score_module,
        embedding_module = aggregation_embedding_module
    )

    update_gate_size_list = cal_size_list(2*hidden_size, 1, update_gate_layer_num)
    update_gate_module =  MLP(update_gate_size_list, nn.Identity)
    update_update_module = nn.Identity()
    update_module = globals().get(update_module_name+'Update')(
        gate_module=update_gate_module,
        update_module=update_update_module
    )

    # readout_score_size_list = cal_size_list(in_channels, 1, readout_score_layer_num)
    # readout_score_module = MLP(readout_score_size_list, nn.Identity)
    # readout_embedding_module = nn.Identity()
    # readout_module = globals().get(readout_name+'Aggregation')(
        # score_module = readout_score_module,
        # embedding_module = readout_embedding_module
    # )
    readout_module = lambda x, attention_x, *args, **kwargs: x[attention_x[:,1]==1]

    score_size_list = cal_size_list(hidden_size, 1, confidence_layer_num)
    score_module = MLP(score_size_list, nn.Identity)

    body_module = BodyV2(
        edge_embedding_module=edge_embedding_module,
        aggregation_module=aggregation_module,
        update_module=update_module,
        readout_module=readout_module,
        score_module=score_module
    )

    # head_size_list = cal_size_list(hidden_size, out_channels, head_layer_num)
    # head_module = MLP(head_size_list, nn.Identity)
    head_module = lambda x:-x

    model_type = globals().get(net_name)
    model = model_type(
        embedding_module = embedding_module,
        body_module = body_module,
        readout_module = readout_module,
        head_module = head_module,
        max_iter = None,
    )

    return model

def SimpleBaselineV5(in_channels, edge_channels, out_channels,
             net_name='IterGNNV4',
             hidden_size=4, embedding_layer_num=1,
             edge_embedding_layer_num=1,
             aggregation_name='Max', aggregation_score_layer_num=1,
             update_module_name='Identity', update_gate_layer_num=1,
             readout_name='Attention', readout_score_layer_num=1,
             confidence_layer_num=1,
             head_layer_num=1,
             *args, **kwargs,
             ):

    embedding_size_list = cal_size_list(in_channels, hidden_size, embedding_layer_num)
    embedding_module = MLP(embedding_size_list, nn.Identity)

    # edge_embedding_size_list = cal_size_list(
        # edge_channels+hidden_size
        # , hidden_size, edge_embedding_layer_num
    # )
    # edge_embedding_module = MLP(edge_embedding_size_list, nn.Identity)
    edge_embedding_module = lambda x:x[edge_channels:]-x[:edge_channels]

    aggregation_score_size_list = cal_size_list(
        edge_channels+2*hidden_size,
        1, aggregation_score_layer_num,
    )
    aggregation_score_module = MLP(aggregation_score_size_list, nn.Identity)
    aggregation_embedding_module = nn.Identity()
    aggregation_module = globals().get(aggregation_name+'Aggregation')(
        score_module = aggregation_score_module,
        embedding_module = aggregation_embedding_module
    )

    update_gate_size_list = cal_size_list(2*hidden_size, 1, update_gate_layer_num)
    update_gate_module =  MLP(update_gate_size_list, nn.Identity)
    update_update_module = nn.Identity()
    update_module = globals().get(update_module_name+'Update')(
        gate_module=update_gate_module,
        update_module=update_update_module
    )

    # readout_score_size_list = cal_size_list(in_channels, 1, readout_score_layer_num)
    # readout_score_module = MLP(readout_score_size_list, nn.Identity)
    # readout_embedding_module = nn.Identity()
    # readout_module = globals().get(readout_name+'Aggregation')(
        # score_module = readout_score_module,
        # embedding_module = readout_embedding_module
    # )
    readout_module = lambda x, attention_x, *args, **kwargs: x[attention_x[:,1]==1]

    score_size_list = cal_size_list(hidden_size, 1, confidence_layer_num)
    score_module = MLP(score_size_list, nn.Identity)

    body_module = BodyV2(
        edge_embedding_module=edge_embedding_module,
        aggregation_module=aggregation_module,
        update_module=update_module,
        readout_module=readout_module,
        score_module=score_module
    )

    # head_size_list = cal_size_list(hidden_size, out_channels, head_layer_num)
    # head_module = MLP(head_size_list, nn.Identity)
    head_module = lambda x:-x

    model_type = globals().get(net_name)
    model = model_type(
        embedding_module = embedding_module,
        body_module = body_module,
        readout_module = readout_module,
        head_module = head_module,
        max_iter = None,
    )

    return model
