import numpy as np
from .array import Array
from robot.utils.rot6d import rmul, inv, rmat


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
