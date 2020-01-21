import numpy as np
from .array import Array
from robot.utils.rot6d import rmul, inv, rmat
from robot.utils.quaternion import qmul, qrot, qinv


class Angular6d(Array):
    def __init__(self, *args, **kwargs):
        super(Angular6d, self).__init__(*args, **kwargs)
        assert self.shape[-1] == 6, "the shape of the tuple should be (xxx, 6)"

    def sample(self):
        return self.np_random.random(size=self.shape)

    def add(self, a, b, scene=None):
        return rmul(a, b)

    def sub(self, a, b, scene=None):
        return rmul(a, inv(b))

    def metric(self, a, scene=None, is_batch=False):
        diff = rmat(a)
        xx = 0.5 * (diff[..., 0, 0] + diff[..., 1, 1] + diff[..., 2, 2] - 1)
        out = (xx - 1) ** 2

        if is_batch:
            out = out.reshape(out.shape[0], -1)
        else: out = out.reshape(-1)
        return out.sum()

    def __repr__(self):
        return "Angular6d(" + str(self.shape) + ")"


class Quaternion(Array):
    def __init__(self, *args, **kwargs):
        super(Quaternion, self).__init__(*args, **kwargs)
        assert self.shape[-1] == 4, "the shape of the quaternion should be (xxx, 4)"

    def sample(self):
        return self.np_random.random(size=self.shape)

    def add(self, a, b, scene=None):
        return qmul(a, b)

    def sub(self, a, b, scene=None):
        return qmul(a, qinv(b))

    def metric(self, a, scene=None, is_batch=False):
        raise NotImplementedError
        diff = rmat(a)
        xx = 0.5 * (diff[..., 0, 0] + diff[..., 1, 1] + diff[..., 2, 2] - 1)
        out = (xx - 1) ** 2

        if is_batch:
            out = out.reshape(out.shape[0], -1)
        else: out = out.reshape(-1)
        return out.sum()

    def __repr__(self):
        return "Angular6d(" + str(self.shape) + ")"
