import numpy as np
import torch
from robot.envs.sapien.exp.utils import set_state
from robot.utils.data_parallel import DataParallel


class NumpyWeightNetwork:
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

        self.in_feature = in_feature
        self.tanh = np.tanh
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

    def __call__(self, x, weights):
        for l in range(self.num_layer):
            i, o, tanh = self.layers[l]
            w = weights[self.w[l]]
            b = weights[self.b[l]]

            w = w.reshape(i, o)
            b = b.reshape(1, o)
            x = x @ w + b
            if tanh:
                x = self.tanh(x)
        return x


class Rollout:
    def __init__(self, make, env_name, num_layer=2, mid_channel=32):
        self.env = make(env_name).unwrapped  # discard the other things...
        inp_dim = self.env.observation_space.shape[0]
        oup_dim = self.env.action_space.shape[0]
        self.network = NumpyWeightNetwork(inp_dim, oup_dim, num_layer, mid_channel)
        self.lb = self.env.action_space.low
        self.ub = self.env.action_space.high

    def __call__(self, s, a, timestep):
        rewards = []
        obs = []
        for s, a, timestep in zip(s, a, timestep):
            set_state(self.env, s)
            s = self.env._get_obs()

            if len(a.shape) > 1:
                assert a.shape[0] == timestep

            reward = 0
            for i in range(timestep):
                if len(a.shape)>1:
                    action = a[i]
                else:
                    action = a

                action = self.network(s, action)
                action = np.maximum(np.minimum(action, self.ub), self.lb)
                s, r, done, _ = self.env.step(action)
                reward += r

                if done:
                    break

            rewards.append(reward)
            obs.append(s)
        return np.array(obs), np.array(rewards)


class DynamicModel:
    def __init__(self, make, env_name, n=20, *args, **kwargs):
        self.model = DataParallel(n, Rollout, make, env_name, *args, **kwargs)

    def __call__(self, s, a, timestep):
        is_cuda = isinstance(a, torch.Tensor)
        device = 'cpu'
        if is_cuda:
            device = s.device
            s = s.detach().cpu().numpy()
            a = a.detach().cpu().numpy()
        obs, r = self.model(s, a, timestep)
        if is_cuda:
            obs = torch.tensor(obs, dtype=torch.float, device=device)
            r = torch.tensor(r, dtype=torch.float, device=device)
        return obs, r
