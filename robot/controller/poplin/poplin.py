# poplin: use neural network to help search the actions (unlike cem)
import torch
from torch import nn
import numpy as np
import copy
from robot.utils import as_input
from robot.utils import AgentBase
from robot.utils.normalizer import Normalizer
from robot.controller.cem import CEM


def normc_init(shape, device, std=1.):
    out = np.random.randn(*shape).astype(np.float32)
    out *= 1. / np.sqrt(np.square(out).sum(axis=0, keepdims=True)) * std
    return torch.tensor(out, device=device, dtype=torch.float)


class WeightNetwork:
    def __init__(self, in_feature, out_feature, num_layers, mid_channels, device='cuda:0'):
        layers = []
        if num_layers == 1:
            layers.append((in_feature, out_feature, False))
        else:
            if isinstance(mid_channels, int):
                mid_channels = [mid_channels] * (num_layers-1)
            assert len(mid_channels) == num_layers - 1
            layers.append((in_feature, mid_channels[0], True))
            for i in range(num_layers-2):
                layers.append((mid_channels[i], mid_channels[i+1], True))
            layers.append((mid_channels[num_layers-2], out_feature, False))

        self.tanh = nn.Tanh()
        self.num_layer = num_layers
        self.layers = layers
        self._device = device

        start = 0
        self.w = []
        self.b = []
        self.scale = []

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
        self.scale = []
        print("WARNING>>>>>> TRICKS on network weights...")
        for idx, (i, o, _) in enumerate(self.layers):
            std = 1. if idx != self.num_layer - 1 else 0.01
            w = normc_init((i, o), device=self._device, std=std)
            b = torch.zeros((o,), device=self._device) * 0
            print(w.mean(), w.std())
            weights += [w.reshape(-1), b.reshape(-1)]

            self.scale.append(std)

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

            #print(x.matmul(w)[:2])
            x = x.matmul(w) * self.scale[l] + b # TODO: add trick here, I don't know the original implementation...
            if tanh:
                x = self.tanh(x)
        return x



class PoplinController(AgentBase):
    # Poplin-P
    # very slow??
    def __init__(self, model, prior, horizon,
                 inp_dim, oup_dim,
                 std=0.1 ** 0.5,
                 replan_period=1,
                 num_layers=2, mid_channels=32,
                 iter_num=5, num_mutation=500, num_elite=100,
                 action_space = None,
                 device='cuda:0', **kwargs):
        self.horizon = horizon
        self.replan_period = replan_period
        self._device = device

        self.network = WeightNetwork(inp_dim, oup_dim, num_layers=num_layers, mid_channels=mid_channels)
        self.normalizer = Normalizer((inp_dim,)).to(device)
        self.cem = CEM(self.rollout, iter_num=iter_num, num_mutation=num_mutation, num_elite=num_elite,
                       std= std,
                       **kwargs)

        self.prior = prior
        self.model = model # model is a forward model

        self.w_buf = None
        self.prev_weights = None
        self.cur_weights = None

        self.weights_dataset = []
        self.obs_dataset = []

        self.lb = self.up = None
        if action_space is not None:
            self.lb = torch.tensor(action_space.low, device=device, dtype=torch.float)
            self.ub = torch.tensor(action_space.high, device=device, dtype=torch.float)


    def network_control(self, obs, weights):
        obs = self.prior.encode(obs)
        obs = self.normalizer(obs)

        out = self.network(obs, weights)
        if self.lb is not None:
            out = torch.max(torch.min(out, self.ub), self.lb)
        return out


    def rollout(self, obs, weights):
        # obs (1, dx)
        # weights (500, w)
        # return rewards

        obs = obs.expand(weights.shape[0], -1) # (500, x)

        reward = 0
        for i in range(weights.shape[1]):
            # in (500, 1, x) out ideally (5, 500, x)
            action = self.network_control(obs[:, None], weights[:, i])[:, 0]
            t, _ = self.model.forward(obs, action) # NOTE that
            if len(t.shape) == 3:
                t = t.mean(dim=0) # mean
            reward = self.prior.cost(obs, action, t) + reward
            obs = t
        return reward

    def update(self, buffer):
        print('fit policy normalizer...')
        if len(self.obs_dataset) == 0:
            data_gen = buffer.make_sampler('fix', 'train', 1, use_tqdm=False)
            for s, _, _ in data_gen:
                self.normalizer.update(self.prior.encode(s))
        else:
            self.normalizer.update(self.prior.encode(torch.stack(self.obs_dataset)))

        print('policy normalizer:')
        print(self.normalizer.mean, self.normalizer.std)

        if len(self.weights_dataset) > 0:
            print('UPDATE weights....')
            self.cur_weights = torch.mean(torch.stack(self.weights_dataset), dim=0)
            print(self.cur_weights[-8:])
        self.weights_dataset = []
        self.obs_dataset = []


    def init_weight(self, horizon):
        # mean
        if self.cur_weights is None:
            self.cur_weights = self.network.init_weights()
        return self.cur_weights[None, :].expand(horizon, -1) # the second dimension is the time


    def reset(self):
        # random sample may be not good
        self.prev_weights = self.init_weight(self.horizon)
        self.w_buf = None


    @as_input(2)
    def __call__(self, obs):
        assert len(obs.shape) == 2 and obs.shape[0] == 1
        if self.w_buf is not None:
            if self.w_buf.shape[0] > 0:
                weight, self.w_buf = self.w_buf[0], self.w_buf[1:]
                self.weights_dataset.append(weight)
                self.obs_dataset.append(obs.detach())
                out = self.network_control(obs, weight)
                return out

        self.prev_weights = self.cem(obs, self.prev_weights)
        self.w_buf, self.prev_weights = torch.split(self.prev_weights, [self.replan_period, self.horizon-self.replan_period])
        self.prev_weights = torch.cat((self.prev_weights, self.init_weight(self.replan_period)))
        return self.__call__(obs)


    def set_model(self, model):
        self.model = model


if __name__ == '__main__':
    network = PoplinController(None, None, None, 10, 3)

    weight = network.network.init_weights()
    x = torch.zeros((100, 10)).cuda()

    network.network(x, weight)
