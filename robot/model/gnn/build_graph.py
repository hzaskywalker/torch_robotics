#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

from .utils import append_position, get_build_graph_func

class Image2Graph(nn.Module):
    def __init__(self, graph_name='fully'):
        super(Image2Graph, self).__init__()
        self.build_graph_func = get_build_graph_func(graph_name)
    def forward(self, data):
        img_list, y_list = append_position(data.x).unbind(dim=0), data.y.unbind(dim=0)
        data_list = [
            self.build_graph_func(
                img.reshape([-1, img.size(-1)]), y.reshape([1,-1])
            )
            for img, y in zip(img_list, y_list)
        ]
        return Batch.from_data_list(data_list).to(data.x.device)



