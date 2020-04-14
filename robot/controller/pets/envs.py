import numpy as np
from robot.envs.toy.plane import GoalPlane
from gym.wrappers import TimeLimit
from robot.envs.gym.half_cheetah import GoalHalfCheetahEnv
import torch
from robot.envs.hyrule.rl_env import ArmReachWithXYZ

class Plane:
    inp_dim = 2
    oup_dim = 2
    @classmethod
    def encode_obs(cls, s):
        return s

    @classmethod
    def add_state(cls, s, delta):
        return s + delta

    @classmethod
    def compute_reward(cls, s, a, t, g):
        return -(((t-g) ** 2).sum(dim=-1)) ** 0.5

class ARMReachInfo:
    inp_dim = 29
    oup_dim = 29 # 直接加上么？

    @classmethod
    def encode_obs(cls, s):
        return s

    @classmethod
    def add_state(cls, s, delta):
        return s + delta

    @classmethod
    def compute_reward(cls, s, a, t, g):
        while len(g.shape) < len(t.shape):
            g = g[None,:]
        return -(((t[..., -3:]-g) ** 2).sum(dim=-1)) ** 0.5


class HalfCheetahPrior:
    inp_dim = 18
    oup_dim = 18

    @classmethod
    def encode_obs(cls, obs):
        return torch.cat([
            obs[..., 1:2],
            obs[..., 2:3].sin(),
            obs[..., 2:3].cos(),
            obs[..., 3:]
        ], dim=-1)

    @classmethod
    def add_state(self, obs, pred):
        assert isinstance(obs, torch.Tensor)
        return torch.cat([
            pred[..., :1],
            obs[..., 1:] + pred[..., 1:]
        ], dim=-1)

    @classmethod
    def obs_cost_fn(cls, obs):
        return -obs[..., 0]

    @classmethod
    def ac_cost_fn(cls, acs):
        return 0.1 * (acs ** 2).sum(dim=-1)

    @classmethod
    def compute_reward(cls, s, a, t, g):
        return -(cls.obs_cost_fn(s) + cls.ac_cost_fn(a))


DICT = {
    'plane': Plane,
    'armreach': ARMReachInfo,
    'cheetah': HalfCheetahPrior
}


def make(env_name):
    if env_name == 'plane':
        return TimeLimit(GoalPlane(), 50), DICT[env_name]
    elif env_name == 'arm':
        return TimeLimit(ArmReachWithXYZ(), 100), None
    elif env_name == 'cheetah':
        return TimeLimit(GoalHalfCheetahEnv(), 1000), DICT[env_name]
    else:
        raise NotImplementedError

