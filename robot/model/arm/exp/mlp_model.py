import torch
from torch import nn
from robot import A

class MLP_ACROBAT(nn.Module):
    def __init__(self, inp_dim, oup_dims, num_layers, mid_channels, batchnorm=False):
        nn.Module.__init__(self)

        self.dof = inp_dim[0]
        self.mlp1 = A.models.MLP(inp_dim[0] * 2+ inp_dim[1] + inp_dim[2],
                                 inp_dim[0] + inp_dim[1], num_layers, mid_channels, batchnorm=batchnorm)

        self.mlp2 = A.models.MLP(inp_dim[0] * 2, oup_dims, num_layers, mid_channels, batchnorm=batchnorm)

    def wrap(self, q):
        # the wrapper is the information of the neural network
        return torch.cat((torch.sin(q), torch.cos(q)), dim=-1)

    def forward(self, state, action):
        q, dq = state[..., :self.dof], state[..., self.dof:]

        inp = torch.cat((self.wrap(q), dq, action), dim=-1)
        delta = self.mlp1(inp) # should we just use add here
        new_q = q + delta[..., :self.dof]
        new_dq = dq + delta[..., self.dof:]
        return torch.cat((new_q, new_dq), dim=-1), self.mlp2(self.wrap(new_q))


