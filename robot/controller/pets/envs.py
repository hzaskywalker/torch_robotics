import numpy as np
from robot.envs.toy.plane import GoalPlane

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
        return -np.linalg.norm(s-t)


DICT = {
    'plane': Plane
}


def make(env_name):
    if env_name == 'plane':
        return GoalPlane(), DICT[env_name]

