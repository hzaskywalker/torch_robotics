import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, relu=False, batch_norm=False, *args, **kwargs):
    models = [layer_init(nn.Conv2d(in_channels, out_channels,
                                   kernel_size, stride, padding=padding, *args, **kwargs))]
    if relu:
        models.append(nn.ReLU())
    if batch_norm:
        models.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*models)


def conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, relu=False, batch_norm=False, *args, **kwargs):
    models = [layer_init(nn.Conv1d(in_channels, out_channels,
                                   kernel_size, stride, padding=padding, *args, **kwargs))]
    if relu:
        models.append(nn.ReLU())
    if batch_norm:
        models.append(nn.BatchNorm1d(out_channels))
    return nn.Sequential(*models)


class fc(nn.Module):
    def __init__(self, in_channels, out_channels, relu=False, batch_norm=False, tanh=False):
        nn.Module.__init__(self)
        models = [nn.Linear(in_channels, out_channels)]
        if relu:
            models.append(nn.ReLU())
        if tanh:
            models.append(nn.Tanh())
        if batch_norm:
            models.append(nn.BatchNorm1d(out_channels))
        self.models = nn.Sequential(*models)

    def forward(self, x):
        batch_size = None
        if len(x.shape) == 3:
            batch_size = x.size(0)
            x = x.view(x.size(0) * x.size(1), -1)
        x = self.models(x)
        if batch_size is not None:
            x = x.view(batch_size, -1, x.size(-1))
        return x


def deconv(in_channels, out_channels, kernel_size, stride=1, padding=0, relu=False, batch_norm=False, *args, **kwargs):
    models = [layer_init(nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size, stride, padding=padding, *args, **kwargs))]
    if relu:
        models.append(nn.ReLU())
    if batch_norm:
        models.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*models)

class Identity(nn.Module):
    def forward(self, x):
        return x

class BatchReshape(nn.Module):
    def __init__(self, *args):
        nn.Module.__init__(self)
        if isinstance(args[0], tuple):
            assert len(args) == 1
            self.shape = args[0]
        else:
            self.shape = args

    def forward(self, inp):
        return inp.view(inp.size(0), *self.shape)
