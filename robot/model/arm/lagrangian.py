# just a test now
# need to write the gradient to gradient explicitly ..., not very hard I think..
import torch
from torch import nn
import numpy as np

from scipy.stats import truncnorm
truncnorm = truncnorm(-2, 2)

def truncated_normal(size, std):
    trunc = truncnorm.rvs(size=size) * std
    return torch.tensor(trunc, dtype=torch.float32)

class grad_fc(nn.Module):
    # fully connnected layer that can support calculate the gradient to its input
    def __init__(self, in_features, out_features):
        super(grad_fc, self).__init__()
        w = truncated_normal(size=(in_features, out_features),
                             std=1.0 / (2.0 * np.sqrt(in_features)))

        self.w = nn.Parameter(w)
        self.b = nn.Parameter(torch.zeros(1, out_features, dtype=torch.float32))

    def forward(self, inp):
        return inp @ self.w + self.b

    def get_jacobian(self, inp, jacobian):
        return jacobian @ self.w.T

class grad_relu(nn.Module):
    def __init__(self):
        super(grad_relu, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, inp):
        return self.relu(inp)

    def get_jacobian(self, inp, jacobian):
        return jacobian * (inp>0).float()

class grad_softplus(nn.Module):
    def __init__(self):
        super(grad_softplus, self).__init__()
        self.softplus = nn.Softplus()

    def forward(self, inp):
        return self.softplus(inp)

    def get_jacobian(self, inp, jacobian):
        return jacobian * torch.sigmoid(inp)


class grad_sequential(nn.Sequential):
    def forward(self, input):
        self.__buffer = []
        for module in self:
            self.__buffer.append([module, input])
            input = module(input)
        return input


    def get_jacobian(self, inp, jacobian):
        # pass
        assert inp is self.__buffer[0][1]
        for module, inp in self.__buffer[::-1]:
            jacobian = module.get_jacobian(inp, jacobian)
        self.__buffer = None
        return jacobian

"""
w = torch.rand(100, 8, device='cuda:0', requires_grad=True)
b = torch.rand(8, device='cuda:0', requires_grad=True)

inp = torch.rand(100, 100, device='cuda:0', requires_grad=True)
x = (inp @ w + b)
musk = torch.ones_like(w)

grad = torch.autograd.grad(x.sum(), w,
        create_graph=True,
        retain_graph=True)
print(grad)
"""
