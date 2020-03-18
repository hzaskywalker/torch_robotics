import torch
from torch import nn
from robot.utils.models import fc

class MLP(nn.Module):
    def __init__(self, inp_dim, oup_dim, num_layers, mid_channels, batchnorm=False):
        nn.Module.__init__(self)

        self.inp_dim = inp_dim
        self.oup_dim = oup_dim

        models = []
        cur = inp_dim
        for i in range(num_layers-1):
            models.append(fc(cur, mid_channels, relu=True, batch_norm=batchnorm))
            cur = mid_channels
        models.append(fc(cur, oup_dim, relu=False))
        self.main = nn.Sequential(*models)

    def forward(self, q):
        return self.main(q)

class MLP_ARM(nn.Module):
    def __init__(self, inp_dim, oup_dims, num_layers, mid_channels, batchnorm=False):
        nn.Module.__init__(self)

        self.q_dim = inp_dim[0]
        self.mlp1 = MLP(inp_dim[0] +inp_dim[1], oup_dims[0], num_layers, mid_channels, batchnorm=batchnorm)
        self.mlp2 = MLP(inp_dim[0]//2, oup_dims[1], num_layers, mid_channels, batchnorm=batchnorm)

    def forward(self, state, action):
        #print(state.shape, action.shape)
        #exit(0)
        new_state = state + self.mlp1(torch.cat((state, action), dim=-1)) # should we just use add here
        return new_state, self.mlp2(new_state[...,:new_state.shape[-1]//2])


def make_model(info, args):
    if args.model == 'mlp':
        model = MLP_ARM(info.inp_dim, info.oup_dim, 4, 256, batchnorm=args.batchnorm)
    else:
        raise NotImplementedError
    return model
