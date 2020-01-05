# poplin: use neural network to help search the actions (unlike cem)
import torch
from torch import nn
import numpy as np
from robot.utils.models import fc
from robot.utils import AgentBase
from robot.utils.normalizer import Normalizer
from robot.controller.cem import CEM


def normc_init(shape, device):
    out = np.random.randn(*shape).astype(np.float32)
    out *= 1. / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
    return torch.tensor(out, device=device, dtype=torch.float)


class WeightNetwork(nn.Module):
    def __init__(self, in_feature, out_feature, num_layers, mid_channels, device='cuda:0'):
        super(WeightNetwork, self).__init__()
        layers = []
        if num_layers == 1:
            layers.append((in_feature, out_feature, False))
        else:
            layers.append((in_feature, mid_channels, True))
            for i in range(num_layers-2):
                layers.append((mid_channels, mid_channels, True))
            layers.append((mid_channels, out_feature, False))

        self.tanh = nn.Tanh()
        self.num_layer = num_layers
        self.layers = layers
        self._device = device

        start = 0
        self.w = []
        self.b = []

        for i, o, _ in self.layers:
            r = start + i * o
            self.w.append(slice(start, r))
            start = r

            r = start + o
            self.b.append(slice(start, r))
            start = r

        print(self.w)
        print(self.b)

    def init_weights(self):
        weights = []
        for i, o, _ in self.layers:
            w = normc_init((i, o), device=self._device)
            b = normc_init((o,), device=self._device)
            weights += [w.reshape(-1), b.reshape(-1)]
        return torch.cat(weights)

    def __call__(self, x, weights):
        for l in range(self.num_layer):
            i, o, tanh = self.layers[l]
            if weights.dim() > 1:
                w = weights[..., self.w[l]]
                b = weights[..., self.b[l]]
            else:
                w = weights[self.w[l]]
                b = weights[self.b[l]]

            w = w.reshape(*weights.shape[:-1], i, o)
            b = b.reshape(*weights.shape[:-1], 1, o)
            x = x.matmul(w) + b
            if tanh:
                x = self.tanh(x)
        return x



class PoplinController(AgentBase):
    # Poplin-P
    # very slow??
    def __init__(self, model, prior, inp_dim, oup_dim, num_layers=3, mid_channels=64,
                 *args, **kwargs):
        self.network = WeightNetwork(inp_dim, oup_dim, num_layers=num_layers, mid_channels=mid_channels)
        self.normalizer = Normalizer(inp_dim)
        self.cem = CEM(self.rollout, *args, **kwargs)

        self.prior = prior
        self.model = model # model is a forward model

    def rollout(self, obs, weights):
        # obs (500, x)
        # weights (500, w)
        # return rewards

        reward = 0
        for i in range(weights.shape[1]):
            action = self.network(obs, weights[:, i]) # ideally (5, 500, x)
            t, _ = self.model(obs, action)
            if len(t.shape) == 3:
                t = t.mean(dim=0) # mean
            reward = self.prior.cost(obs, action, t) + reward
            obs = t
        return reward


    def update(self, buffer):
        # update with past trajectories
        # directly copy the forward model's normalizer...
        if 'obs_norm' in self.model.__dir__():
            # use the same normalizer as the model
            self.normalizer = self.model.normalizer.copy()
        else:
            raise NotImplementedError

        pass


    def init_weight(self):
        pass


    def __call__(self, obs):
        pass


if __name__ == '__main__':
    net = PoplinController(None, None, 25, 8).network

    import tqdm

    weight = net.init_weights()
    ans = 0
    for i in tqdm.trange(30 * 5 * 100):
        x = torch.rand((500, 1, 25), device='cuda:0', dtype=torch.float)
        weights = torch.rand((500, weight.shape[0]), device='cuda:0', dtype=torch.float)
        ans = net(x, weights) + ans
        print((ans[1] - net(x[1], weights[1])).abs().max())
        print(ans.shape)
        exit(0)
