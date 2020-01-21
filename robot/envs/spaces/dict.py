import numpy as np
import torch
from collections import OrderedDict
from .utils import to_numpy, to_tensor, serialize, cat
from .space import Type, Space

class Dict(OrderedDict, Type):
    def __init__(self, *args, is_batch=False, **kwargs):
        OrderedDict.__init__(self, *args, **kwargs)
        self.is_batch = is_batch

    def serialize(self):
        out = []
        for i, v in self.items():
            out.append(serialize(v, self.is_batch))
        return cat(out, dim=-1)

    def numpy(self):
        return Dict([(i, to_numpy(v)) for i, v in self.items()], is_batch=self.is_batch)

    def tensor(self, device='cuda:0'):
        return Dict([(i, to_tensor(v, device)) for i, v in self.items()], is_batch=self.is_batch)

    def to(self, device='cuda:0'):
        for i, v in self.items():
            self[i] = to_tensor(v, device)
        return self

    def __add__(self, other):
        out = Dict()
        for i in self:
            out[i] = self[i] + other[i]
        return Dict(out, is_batch=self.is_batch)

    def __sub__(self, other):
        out = Dict()
        for i in self:
            out[i] = self[i] - other[i]
        return Dict(out, is_batch=self.is_batch)

    def __index__(self, index):
        raise NotImplementedError

    def __repr__(self):
        return "Dict(" + ", ". join([str(k) + ":" + str(s) for k, s in self.items()]) + ")"

    def metric(self):
        ans = 0
        for i, v in self.items():
            ans = ans + v.metric()
        return ans

    def id(self, index):
        return Dict([(i, v.id(index)) for i, v in self.items()])

    @property
    def shape(self):
        return OrderedDict([(i, v.shape) for i, v in self.items()])

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        else:
            try:
                return self[item]
            except KeyError:
                raise AttributeError

    def __setattr__(self, key, value):
        if key not in self:
            self.__dict__[key] = value
        else:
            self[key] = value


class DictSpace(OrderedDict, Space):
    def seed(self, seed=None):
        [space.seed(seed) for space in self.values()]

    def sample(self):
        return Dict([(k, space.sample()) for k, space in self.items()])

    def contains(self, x):
        if not isinstance(x, dict) or len(x) != len(self):
            return False
        for k, space in self.items():
            if k not in x:
                return False
            if not space.contains(x[k]):
                return False
        return True

    def __getitem__(self, key):
        return self[key]

    def __repr__(self):
        return "DictSpace(" + ", ". join([str(k) + ":" + str(s) for k, s in self.items()]) + ")"

    def __contains__(self, item):
        # override the in
        return self.contains(item)

    def from_numpy(self, data, is_batch=False):
        l = 0
        out = Dict()
        for i, spec in self.items():
            s = spec.size
            d = data[l:l+s] if not is_batch else data[:, l:l+s]
            out[i] = spec.from_numpy(d, is_batch)
            l += s
        out.is_batch = is_batch
        return out

    @property
    def size(self):
        return sum([i.size for _, i in self.items()])

