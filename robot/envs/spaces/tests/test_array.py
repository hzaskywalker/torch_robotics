import numpy as np
import torch
from robot.envs.spaces.array import ArraySpace, Array

def check_np_array(a, b):
    return ((a-b)**2).sum() < 1e-6

def test_array_space():
    space = ArraySpace(np.array([1, 1, 1]))
    a = space.sample()

    # numpy
    out = a.data
    assert check_np_array(np.array(a), out), "numpy error"
    assert check_np_array(np.array(a.tensor('cpu')),  out), "cpu tensor error"
    assert check_np_array(np.array(a.tensor('cuda:0')),  out), "cuda:0 tensor error"

    # cuda
    a = a.tensor('cuda:0')
    assert isinstance(a.data, torch.Tensor)
    assert check_np_array(np.array(a), out) #cuda

    # operator
    a = Array(np.array([1, 2, 3, 5]))
    b = Array(np.array([2, 3, 4, 9]))

    assert check_np_array(np.array(a+b), np.array([3, 5, 7, 14]))
    assert check_np_array(np.array(a-b), np.array([-1, -1, -1, -4]))
    assert np.abs(a.metric() - (1**2 + 2**2 + 3**2 + 5**2)) < 1e-5

    # batch version
    a = Array(np.array([[1, 2, 3], [2, 3, 4]]), is_batch=True)
    assert check_np_array(a.metric(), np.array([1**2 + 2**2 + 3**2, 2**2 + 3** 2 + 4**2]))

    # in
    assert Array(np.array([0.1, 0.1, 0.1])) in space
    assert Array(np.array([-0.1, -0.1, -0.1])) in space
    assert not Array(np.array([-1.1, -0.1, -0.1])) in space


if __name__ == '__main__':
    test_array_space()