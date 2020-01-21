import numpy as np
from .array import Array, ArraySpace
from robot.utils.rot6d import rmul, inv, rmat


class Angular6d(Array):
    def __init__(self, data, is_batch=False):
        assert data.shape[-1] == 6
        super(Angular6d, self).__init__(data, is_batch)

    def __add__(self, other):
        return Angular6d(rmul(self.data, other.data))

    def __sub__(self, other):
        return Angular6d(rmul(self.data, inv(other.data)))

    def metric(self):
        diff = rmat(self.data)
        # cos(theta)
        print(self.data)
        xx = 0.5 * (diff[..., 0, 0] + diff[..., 1, 1] + diff[..., 2, 2] - 1)
        out = (xx - 1) ** 2

        if self.is_batch:
            out = out.reshape(out.shape[0], -1)
        else: out = out.reshape(-1)
        return out.sum()

    def __repr__(self):
        return "Angular6d(" + str(self.data) + ")"


class Angular6dSpace(ArraySpace):
    def __init__(self, shape):
        shape = shape + (6,) # add 6 dimension at the end
        super(Angular6dSpace, self).__init__(low=-1, high=1, shape=shape)
        self.cls = Angular6d

    def sample(self):
        # how to sample a random rotation in three d
        #raise NotImplementedError
        # TODO: I don't know how to do it correctly without sampling a quaternion first
        return Angular6d(self.np_random.random(size=self.shape))

    def __repr__(self):
        return "Angular6d(" + str(self.shape) + ")"
