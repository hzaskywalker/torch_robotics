import torch
import numpy as np
from .space import Type, Space
from .utils import to_numpy, to_tensor

TYPE_DICT = {
    np.dtype('float32'): torch.float,
    np.dtype('int64'): torch.long,
}


class Array(Type):
    def __init__(self, data, is_batch=False):
        assert isinstance(data, torch.Tensor) or isinstance(data, np.ndarray)
        super(Array, self).__init__(data, is_batch)

    def numpy(self):
        return to_numpy(self.data, self.is_batch)

    def tensor(self, device='cuda:0'):
        return to_tensor(self.data, device)

    @classmethod
    def from_numpy(cls, data, shape, is_batch=False):
        shape = (-1,) + shape if is_batch else shape
        return Array(data.reshape(shape), is_batch)

    @classmethod
    def from_tensor(cls, data, shape, is_batch=False):
        # the function is the same...
        return cls.from_numpy(data, shape, is_batch)

    def __add__(self, other):
        return Array(self.data + other.data)

    def __sub__(self, other):
        return Array(self.data - other.data)

    def id(self, index):
        return Array(self.data[index], isinstance(index, tuple)) #if it's tuple, it's still a batch, otherwise a tuple

    def __repr__(self):
        return "Array("+str(self.data)+")"

    def metric(self):
        d = self.data.reshape(-1) if not self.is_batch else self.data.reshape(self.data.shape[0], -1)
        return (d**2).sum(-1)

    @property
    def shape(self):
        return self.data.shape


def _get_precision(dtype):
    if np.issubdtype(dtype, np.floating):
        return np.finfo(dtype).precision
    else:
        return np.inf

class ArraySpace(Space):
    def __init__(self, low, high=None, shape=None, dtype=np.float32):
        if high is None:
            low, high = -low, low

        if shape is None:
            assert low.shape == high.shape, "box dimension mismatch"
            shape = low.shape
        else:
            assert np.isscalar(low) and np.isscalar(high)
            low = np.zeros(shape, dtype=np.int64) + low
            high = np.zeros(shape, dtype=np.int64) + high

        self.low, self.high = low, high
        self.bounded_below = -np.inf < self.low
        self.bounded_above = np.inf > self.high

        super(ArraySpace, self).__init__(shape, dtype, Array)

    def is_bounded(self, manner="both"):
        below = np.all(self.bounded_below)
        above = np.all(self.bounded_above)
        if manner == "both":
            return below and above
        elif manner == "below":
            return below
        elif manner == "above":
            return above
        else:
            raise ValueError("manner is not in {'below', 'above', 'both'}")


    def sample(self):
        """
                Generates a single random sample inside of the Box.
                In creating a sample of the box, each coordinate is sampled according to
                the form of the interval:

                * [a, b] : uniform distribution
                * [a, oo) : shifted exponential distribution
                * (-oo, b] : shifted negative exponential distribution
                * (-oo, oo) : normal distribution
                """
        high = self.high if self.dtype.kind == 'f' \
            else self.high.astype('int64') + 1
        sample = np.empty(self.shape)

        # Masking arrays which classify the coordinates according to interval
        # type
        unbounded = ~self.bounded_below & ~self.bounded_above
        upp_bounded = ~self.bounded_below & self.bounded_above
        low_bounded = self.bounded_below & ~self.bounded_above
        bounded = self.bounded_below & self.bounded_above

        # Vectorized sampling by interval type
        sample[unbounded] = self.np_random.normal(
            size=unbounded[unbounded].shape)

        sample[low_bounded] = self.np_random.exponential(
            size=low_bounded[low_bounded].shape) + self.low[low_bounded]

        sample[upp_bounded] = -self.np_random.exponential(
            size=upp_bounded[upp_bounded].shape) + self.high[upp_bounded]

        sample[bounded] = self.np_random.uniform(low=self.low[bounded],
                                                 high=high[bounded],
                                                 size=bounded[bounded].shape)
        if self.dtype.kind == 'i':
            sample = np.floor(sample)

        return Array(sample.astype(self.dtype))

    def contains(self, x):
        if isinstance(x, list):
            x = np.array(x)  # Promote list to array for contains check
        assert isinstance(x, np.ndarray)
        return x.shape == self.shape and np.all(x >= self.low) and np.all(x <= self.high)

    def __repr__(self):
        return "Array(Shape="+str(self.shape)+")"


class Discrete(Space):
    def __init__(self, low, high=None, shape=None):
        if high is None:
            low, high = low * 0, low

        if shape is not None:
            low = np.zeros(shape, dtype=np.int64) + low
            high = np.zeros(shape, dtype=np.int64) + high

        self.low = low
        self.high = high
        shape = self.low.shape

        super(Discrete, self).__init__(shape, np.int64, Array)

    def sample(self):
        return Array(self.np_random.randint(self.low, self.high))

    def contains(self, x):
        if isinstance(x, list):
            x = np.array(x)
        assert isinstance(x, np.ndarray)
        x = x.astype(self.dtype)
        return x.shape == self.shape and np.all(x >= self.low) and np.all(x < self.high)
