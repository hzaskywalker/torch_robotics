import torch
import numpy as np
from robot.envs.spaces.graph import *

def check_np_array(a, b):
    return ((a-b)**2).sum() < 1e-6


def test():
    n = 5
    m = 8
    space = GraphSpace(n, m, 1)

    a = space.sample()
    assert isinstance(space(a, is_batch=False).__array__(), np.ndarray)

    b = space.deserialize(space.serialize(a, False), is_batch=False)

    out = np.concatenate((
        a['node']['p'].reshape(-1), a['node']['v'].reshape(-1), a['node']['w'].reshape(-1), a['node']['dw'].reshape(-1), a['edge'].reshape(-1), a['graph'].reshape(-1)))
    a = space(a, is_batch=False)
    b = space(b, is_batch=False)

    assert check_np_array(b.numpy().serialize(),  out), "cpu tensor error"

    assert check_np_array(a.tensor('cpu').numpy().serialize(),  out), "cpu tensor error"
    assert check_np_array(a.tensor('cuda:0').numpy().serialize(),  out), "cuda:0 tensor error"

    # cuda
    a = a.tensor('cuda:0')
    assert isinstance(a['node']['p'].data, torch.Tensor)
    assert check_np_array(a.numpy().serialize(), out) #cuda

    # operator
    #a = Array(np.array([1,2,3, 5]))
    #b = Array(np.array([2,3,4, 9]))

    a = space(space.sample(), is_batch=False)
    b = space(space.sample(), is_batch=False)
    assert np.abs((a-a).metric()) < 1e-8
    assert np.abs((a-b).metric()) > 1e-8
    #assert check_np_array((a+b).numpy(), np.array([3, 5, 7, 14]))
    #assert check_np_array((a-b).numpy(), np.array([-1, -1, -1, -4]))
    #assert np.abs(a.metric() - (1**2 + 2**2 + 3**2 + 5**2)) < 1e-5

    # batch version
    #a = Array(np.array([[1,2,3], [2,3,4]]), is_batch=True)
    #assert check_np_array(a.metric(), np.array([1**2 + 2**2 + 3**2, 2**2 + 3** 2 + 4**2]))
    c = np.stack([a.numpy().serialize(), b.numpy().serialize()])
    d = space(space.deserialize(c, is_batch=True), is_batch=True)
    assert check_np_array(d.id(0).numpy().serialize(), a.numpy().serialize())
    assert check_np_array(d.id(1).numpy().serialize(), b.numpy().serialize())
    assert check_np_array(d.numpy().serialize(), c)

    # in
    #assert [0.1, 0.1, 0.1] in spaces
    #assert [-0.1, -0.1, -0.1] in spaces
    #assert not [-1.1, -0.1, -0.1] in spaces


if __name__ == '__main__':
    test()