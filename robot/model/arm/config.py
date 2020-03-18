# configuaration..
import numpy as np
import torch

class ARMINFO:
    max_q = 7
    max_dq = 2000 * 0.1
    max_a = 1

    def __init__(self, env):
        self.inp_dim = (2 * 7, 7)
        self.oup_dim = (2 * 7, 3)  # predict the end effector position...

        env = env.unwrapped
        dofs = env._actuator_dof['agent']
        self.dof_id = np.concatenate((dofs, dofs + 13))

    def compute_reward(self, s, a, t, ee, g):
        while len(g.shape) < len(ee.shape):
            g = g[None, :]
        return -(((ee[..., -3:] - g) ** 2).sum(dim=-1)) ** 0.5

    def encode_obs(self, obs):
        s = obs[..., self.dof_id]
        assert len(self.dof_id) == 14
        s[..., 7:] *= 0.1  # times dt to make it consistent with state
        return s, obs[..., -3:]



class ACROBATINFO:
    def __init__(self, env):
        env = env.unwrapped
        dofs = env._actuator_dof['agent']
        length = env._length
        d = len(length)

        self.inp_dim= (2 * d, d)
        self.oup_dim= (2 * d, 3) # predict the end effector position...
        self.dof_id = np.concatenate((dofs, dofs + d))
        self.DIM = d


    @classmethod
    def compute_reward(cls, s, a, t, ee, g):
        while len(g.shape) < len(t.shape):
            g = g[None,:]
        return -(((ee[..., [-3, -1]] -g) ** 2).sum(dim=-1)) ** 0.5

    def encode_obs(self, s):
        s = s[..., self.dof_id]
        assert len(self.dof_id) == 2 * self.DIM
        s[..., self.DIM:] *= 0.1  # times dt to make it consistent with state
        return s, s[..., -3:]


class PlaneInfo:
    inp_dim = (4, 2)
    oup_dim = (4, 2)

    max_q = 100
    max_dq = 100
    max_a = 1

    def __init__(self, env):
        pass

    def encode_obs(self, s):
        return torch.cat((s, s), dim=-1), s

    def compute_reward(self, s, a, t, ee, g):
        while len(g.shape) < len(ee.shape):
            g = g[None, :]
        return -(((ee-g) ** 2).sum(dim=-1)) ** 0.5


def make_info(env_name, env):
    if env_name == 'arm':
        info = ARMINFO
    elif env_name == 'acrobat2':
        info = ACROBATINFO
    elif env_name == 'plane':
        info = PlaneInfo
    else:
        raise NotImplementedError

    return info(env)
