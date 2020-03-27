import torch
import numpy as np
from torch import nn
from robot import U, tr


class Frame:
    input_dims = None
    output_dims = None

    def as_input(self, action):
        # return a series or a dict of tensors that can be as the input to the network
        raise NotImplementedError

    def as_observation(self):
        raise NotImplementedError

    def add(self, *args):
        raise NotImplementedError

    def calc_loss(self, label):
        raise NotImplementedError

    @classmethod
    def from_observation(cls, observation):
        # build new frame from the observation
        raise NotImplementedError

    @property
    def EE(self):
        # In fact it's the achieved goal
        raise NotImplementedError

    def compute_reward(self, goal):
        # Calculate the reward, currently we only support the form like R(t, g)
        raise NotImplementedError

    def iter(self):
        raise NotImplementedError

    @classmethod
    def new(cls, iter):
        return cls(*iter)

    def __getitem__(self, item):
        assert not isinstance(item, str)
        return self.new([i[item] for i in self.iter()])

    @classmethod
    def stack(cls, list, dim=-2):
        outs = [i.iter() for i in list]
        outs = [torch.stack([j[i] for j in outs], dim=dim) for i in range(len(outs[0]))]
        return cls.new(outs)

    def cpu(self):
        return self.new([U.tocpu(i) for i in self.iter()])

    def cuda(self):
        return self.new([U.togpu(i) for i in self.iter()])


    @property
    def shape(self):
        return [i.shape for i in self.iter()]

    @property
    def size(self):
        return sum([i.size for i in self.iter()])

    def __iter__(self):
        element = self.iter()
        for i in  zip(*element):
            yield self.new(i)



class ArmBase(Frame):
    max_q = 7
    max_dq = 2000 * 0.1
    max_a = 1
    loss = nn.MSELoss()
    dim = 7

    input_dims = (2 * 7, 7)
    output_dims = (2 * 7, 3)

    def __init__(self, q, dq, ee=None):
        self.q = q
        self.dq = dq
        self.ee = ee

    def as_input(self, action):
        action = action.clamp(-self.max_a, self.max_a)
        return torch.cat((self.q, self.dq), dim=-1), action

    def as_observation(self):
        s = self.q
        assert s.shape[0] == 7, f"ERROOR: s.shape: {s.shape}"
        q = np.zeros((*s.shape[:-1], 29))

        q[np.arange(7) + 1] = U.tocpu(s)
        if self.ee is not None:
            q[-3:] = U.tocpu(self.ee)
        return {'observation': q}

    def add(self, new_state, ee):
        q = new_state[...,:self.dim].clamp(-self.max_q, self.max_q)
        dq = new_state[..., self.dim:self.dim * 2].clamp(-self.max_dq, self.max_dq)
        return self.__class__(q, dq, ee)

    def calc_loss(self, label):
        # label are also type a frame..
        assert self.q.shape == label.q.shape
        assert self.dq.shape == label.dq.shape
        assert self.ee.shape == label.ee.shape
        return {
            'q_loss': self.loss(self.q, label.q),
            'dq_loss': self.loss(self.dq, label.dq),
            'ee_loss': self.loss(self.ee, label.ee)
        }

    @classmethod
    def from_observation(cls, observation):
        # make sure observation is a tensor
        # this is indeed the deserialization ...
        q = observation[..., np.arange(7) + 1]
        dq = observation[..., np.arange(7) + 14] * 0.1
        ee = observation[..., -3:]
        return cls(q, dq, ee)

    @property
    def EE(self):
        return self.ee

    def compute_reward(self, goal):
        while len(goal.shape) < len(self.ee.shape):
            goal = goal[None, :]
        return -(((self.ee - goal) ** 2).sum(dim=-1)) ** 0.5

    def iter(self):
        return self.q, self.dq, self.ee


class Plane(ArmBase):
    input_dims = (4, 2)
    output_dims = (4, 2)

    dim = 2
    max_q = 100
    max_dq = 100
    max_a = 1

    @classmethod
    def from_observation(cls, observation):
        return cls(observation, observation * 0, observation)


def make_frame_cls(env_name, env):
    # hopefully we don't need env
    if env_name == 'arm':
        return ArmBase
    elif env_name == 'plane':
        return Plane
    else:
        raise NotImplementedError
