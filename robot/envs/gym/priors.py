import torch
import numpy as np

class StatePrior:
    def encode(self, x):
        return x

    def decode(self, x):
        raise NotImplementedError

    def add(self, s, d):
        return s + d

    def cost(self, s, a, t, it=None):
        return self.obs_cost_fn(t) + self.ac_cost_fn(a)


class CartPolePrior(StatePrior):
    # Create and move this tensor to GPU so that
    # we do not waste time moving it repeatedly to GPU later

    def __init__(self):
        super(CartPolePrior, self).__init__()
        self.inp_dim = 5
        self.oup_dim = 4
        self.ee_sub = torch.tensor([0.0, 0.6], dtype=torch.float)
        self.TASK_HORIZON = 200

    def encode(self, obs):
        if isinstance(obs, np.ndarray):
            return np.concatenate([np.sin(obs[..., 1:2]), np.cos(obs[..., 1:2]), obs[..., :1], obs[..., 2:]], axis=1)
        elif isinstance(obs, torch.Tensor):
            return torch.cat([
                obs[..., 1:2].sin(),
                obs[..., 1:2].cos(),
                obs[..., :1],
                obs[..., 2:]
            ], dim=-1)


    def obs_cost_fn(self, obs):
        ee_pos = self._get_ee_pos(obs)
        ee_pos -= self.ee_sub.to(obs.device)
        ee_pos = ee_pos ** 2

        ee_pos = - ee_pos.sum(dim=-1)

        return - (ee_pos / (0.6 ** 2)).exp()

    def ac_cost_fn(self, acs):
        return 0.01 * (acs ** 2).sum(dim=-1)

    def _get_ee_pos(self, obs):
        x0, theta = obs[..., :1], obs[..., 1:2]

        return torch.cat([
            x0 - 0.6 * theta.sin(), -0.6 * theta.cos()
        ], dim=-1)


class HalfCheetahPrior(StatePrior):
    def __init__(self):
        super(HalfCheetahPrior, self).__init__()
        self.inp_dim = 18
        self.oup_dim = 18
        self.TASK_HORIZON = 1000

    def encode(self, obs):
        if isinstance(obs, np.ndarray):
            return np.concatenate([obs[..., 1:2], np.sin(obs[..., 2:3]), np.cos(obs[..., 2:3]), obs[..., 3:]], axis=-1)
        elif isinstance(obs, torch.Tensor):
            return torch.cat([
                obs[..., 1:2],
                obs[..., 2:3].sin(),
                obs[..., 2:3].cos(),
                obs[..., 3:]
            ], dim=-1)

    def add(self, obs, pred):
        assert isinstance(obs, torch.Tensor)
        return torch.cat([
            pred[..., :1],
            obs[..., 1:] + pred[..., 1:]
        ], dim=-1)

    def obs_cost_fn(self, obs):
        return -obs[..., 0]

    def ac_cost_fn(self, acs):
        return 0.1 * (acs ** 2).sum(dim=-1)


PRIORS = {
    'MBRLCartpole-v0': CartPolePrior(),
    'MBRLHalfCheetah-v0': HalfCheetahPrior()
}
