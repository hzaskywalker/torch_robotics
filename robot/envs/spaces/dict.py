import numpy as np
import torch
from collections import OrderedDict
from .utils import serialize, cat
from .space import Space, Frame


class Dict(OrderedDict, Space):
    def seed(self, seed=None):
        [space.seed(seed) for space in self.values()]

    @property
    def size(self):
        return sum([i.size for _, i in self.items()])

    @property
    def shape(self):
        return OrderedDict([(i, v.shape) for i, v in self.items()])

    def serialize(self, state, is_batch=False):
        out = []
        for i, space in self.items():
            out.append(space.serialize(state[i], is_batch=is_batch))
        return cat(out, dim=-1)
    # TODO: the deserialize code support the spaces.. but not enough

    def deserialize(self, state, is_batch=False):
        l = 0
        out = OrderedDict()
        for i, spec in self.items():
            s = spec.size
            d = state[l:l + s] if not is_batch else state[:, l:l + s]
            out[i] = spec.deserialize(d, is_batch)
            l += s
        return out

    def id(self, state, index):
        #return Frame(self.state[index], isinstance(index, tuple))
        return OrderedDict([(i, self[i].id(v, index)) for i, v in state.items()])

    def sample(self):
        return OrderedDict([(k, space.sample()) for k, space in self.items()])

    def contains(self, x):
        if not isinstance(x, dict) or len(x) != len(self):
            return False
        for k, space in self.items():
            if k not in x:
                return False
            if not space.contains(x[k]):
                return False
        return True

    def observe(self, state, scene=None):
        return state # just return the ordered dict

    def add(self, a, b, scene=None):
        out = OrderedDict()
        for i, spec in self.items():
            out[i] = spec.add(a[i], b[i])
        return out

    def sub(self, a, b, scene=None):
        out = OrderedDict()
        for i, spec in self.items():
            out[i] = spec.sub(a[i], b[i])
        return out

    def __index__(self, index):
        raise NotImplementedError

    def metric(self, a, scene=None, is_batch=False):
        ans = 0
        for i, spec in self.items():
            ans = ans + spec.metric(a[i])
        return ans

    def __repr__(self):
        return "DictSpace(" + ", ". join([str(k) + ":" + str(s) for k, s in self.items()]) + ")"

    def __call__(self, state, is_batch, scene=None):
        assert isinstance(state, OrderedDict)
        # return a Frame variable that help us to write the code...
        return DictFrame(self, state, scene, is_batch)



class DictFrame(Frame):
    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        else:
            try:
                return self.__dict__['state'][item]
            except KeyError:
                raise AttributeError

    def __setattr__(self, key, value):
        if 'state' in self.__dict__:
            if key in self.__dict__['state']:
                self.__dict__['state'][key] = value
                return
        self.__dict__[key] = value

    def __getitem__(self, item):
        return self.state[item]

    def __repr__(self):
        return "DictFrame("+str(self.state)+") of "+str(self.space)

    @property
    def shape(self):
        def get_shape(a):
            if isinstance(a, np.ndarray) or isinstance(a, torch.Tensor):
                return a.shape
            else:
                assert isinstance(a, OrderedDict)
                return OrderedDict([(i, get_shape(v)) for i, v in a.items()])
        return get_shape(self.state)
