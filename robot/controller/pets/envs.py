import numpy as np
from robot.envs.toy.plane import GoalPlane
from gym.wrappers import TimeLimit

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


DICT = {
    'plane': Plane
}


def make(env_name):
    if env_name == 'plane':
        return TimeLimit(GoalPlane(), 50), DICT[env_name]

