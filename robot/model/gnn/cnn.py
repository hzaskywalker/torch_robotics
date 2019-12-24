#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn

from .utils import get_activation_func
from .utils import cal_size_list

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, layer_num=2,
                 activation_name='LeakyReLU', kernel_size=3, stride=1):
        super(CNN, self).__init__()
        self.activation_type = get_activation_func(activation_name)
        conv_dim_list = cal_size_list(in_channels, out_channels, layer_num)
        self.convs = nn.Sequential(
            *(
                nn.Sequential(
                    nn.Conv2d(conv_dim_list[ln], conv_dim_list[ln+1],
                              kernel_size=kernel_size, stride=stride,
                              padding=int((kernel_size-kernel_size%2)/2)),
                    self.activation_type()
                )
                for ln in range(layer_num)
            )
        )
    def forward(self, data):
        img = data.x.permute([0, 3, 1, 2])
        feat = self.convs(img)
        data.x = feat.permute([0, 2, 3, 1])
        return data



