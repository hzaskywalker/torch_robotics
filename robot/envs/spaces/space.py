import numpy as np
import torch
from collections import OrderedDict
from gym.spaces import Space as GymSpace
from .utils import to_numpy, to_tensor, serialize, deserialize


class Space(GymSpace):
    def __init__(self, shape, dtype):
        # property of leaf spaces
        import numpy as np  # takes about 300-400ms to import, so we load lazily
        self._size = None
        self._shape = shape
        self.dtype = None if dtype is None else np.dtype(dtype)
        self.np_random = None
        self.seed()

    @property
    def size(self):
        # return the size of variable in this spaces
        if self._size is None:
            self._size = int(np.prod(self.shape))
        return self._size

    @property
    def shape(self):
        # return the shape of variable in this spaces
        return self._shape

    def serialize(self, state, is_batch=False):
        # return the serialization
        return serialize(state, is_batch)

    def deserialize(self, vector, is_batch=False):
        # return the de-serialization
        return deserialize(vector, self.shape, is_batch)

    def id(self, state, index):
        #return Frame(self.state[index], isinstance(index, tuple))
        if isinstance(state, np.ndarray) or isinstance(state, torch.Tensor):
            return state[index]
        else:
            raise NotImplementedError

    def to_numpy(self, state):
        # return the same type object, while the data is numpy
        return to_numpy(state)

    def to_tensor(self, state, device='cuda:0'):
        # return the same type object, while the data is saved as pytorch Tensor
        return to_tensor(state, device)

    # main function to override for the previous tasks...
    def contains(self, x):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    # main function to override for new
    def observe(self, state, scene=None):
        raise NotImplementedError

    def sub(self, a, b, scene=None):
        raise NotImplementedError

    def add(self, a, b, scene=None):
        raise NotImplementedError

    def metric(self, a, scene=None, is_batch=False):
        raise NotImplementedError

    @property
    def derivative_space(self):
        # the deriviative spaces will be the same as observation_space by default
        return self.observation_space

    @property
    def observation_space(self):
        # the observation spaces will be the same to self by default
        return self

    def __repr__(self):
        return "Space("+str(self.shape)+")"

    def __call__(self, state, is_batch, scene=None):
        # return a Frame variable that help us to write the code...
        return Frame(self, state, scene, is_batch)


class Frame:
    def __init__(self, space: Space, state, scene=None, is_batch=False):
        self.state = state
        self.space = space
        self.scene = scene
        self.is_batch = is_batch

    def __array__(self):
        return self.space.serialize(self.numpy().observe(), self.is_batch) # must be numpy

    def serialize(self):
        return self.space.serialize(self.state, self.is_batch)

    def observe(self):
        return self.space.observe(self.state, self.scene)

    def numpy(self):
        scene = None if self.scene is None else to_numpy(self.scene)
        return self.__class__(self.space, to_numpy(self.state), scene, self.is_batch)

    def tensor(self, device):
        scene = None if self.scene is None else to_tensor(self.scene, device)
        return self.__class__(self.space, to_tensor(self.state, device), scene, self.is_batch)

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            other = other.state
        return self.__class__(self.space, self.space.sub(self.state, other, self.scene), self.scene, self.is_batch)

    def __add__(self, other):
        if isinstance(other, self.__class__):
            other = other.state
        return self.__class__(self.space, self.space.add(self.state, other, self.scene), self.scene, self.is_batch)

    def metric(self):
        return self.space.metric(self.state, self.scene, self.is_batch)

    def id(self, index):
        assert self.is_batch
        return self.__class__(self.space, self.space.id(self.state, index), self.scene, is_batch=isinstance(index, tuple))

    def __repr__(self):
        return "Frame("+str(self.state)+") of "+str(self.space)

