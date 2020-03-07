import numpy as np
from robot.envs.toy.plane import GoalPlane
from gym.wrappers import TimeLimit
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
    def compute_reward(cls, s, t):
        return -(((s-t) ** 2).sum(dim=-1)) ** 0.5

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
    def compute_reward(cls, s, t):
        while len(t.shape) < len(s.shape):
            t = t[None,:]
        return -(((s[..., -3:]-t) ** 2).sum(dim=-1)) ** 0.5


DICT = {
    'plane': Plane,
    'armreach': ARMReachInfo
}


def make(env_name):
    if env_name == 'plane':
        return TimeLimit(GoalPlane(), 50), DICT[env_name]
    elif env_name == 'armreach':
        return TimeLimit(ArmReachWithXYZ(), 50), DICT[env_name]
    else:
        raise NotImplementedError

