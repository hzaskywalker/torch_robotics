import numpy as np
from gym.spaces import Space as GymSpace

class Type(object):
    def __init__(self, data, is_batch=False):
        self.data = data
        self.is_batch = is_batch

    def __array__(self):
        return self.numpy()

    def numpy(self):
        # return a numpy
        raise NotImplementedError

    def tensor(self, device='cuda:0'):
        # return a tensor
        raise NotImplementedError

    def to(self, device='cuda:0'):
        if device == 'numpy':
            self.data = self.numpy()
        else:
            self.data = self.tensor(device)
        return self

    def __add__(self, other):
        raise NotImplementedError

    def __sub__(self, other):
        raise NotImplementedError

    def id(self, index):
        raise NotImplementedError

    def metric(self):
        raise NotImplementedError


class Space(GymSpace):
    def __init__(self, shape=None, dtype=None, cls=None):
        import numpy as np  # takes about 300-400ms to import, so we load lazily
        self.shape = None if shape is None else tuple(shape)
        self.dtype = None if dtype is None else np.dtype(dtype)
        self.cls = np.ndarray if cls is None else cls
        self.np_random = None
        self.seed()

    def from_numpy(self, data, is_batch=False):
        import numpy as np  # takes about 300-400ms to import, so we load lazily
        if self.cls is np.ndarray:
            shape = (-1,) + self.shape if is_batch else self.shape
            return np.adarray(data).reshape(shape)
        else:
            return self.cls.from_numpy(data, self.shape, is_batch)

    def contains(self, x):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    @property
    def size(self):
        return int(np.prod(self.shape))
