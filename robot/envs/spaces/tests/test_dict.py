import numpy as np
import tqdm
import torch
from robot.envs.spaces.dict import DictSpace, Dict
from robot.envs.spaces.array import Array, ArraySpace
from robot.envs.spaces.angular import Angular6d, Angular6dSpace

def check_np_array(a, b):
    return ((a-b)**2).sum() < 1e-6

def test_array_space():
    for i in tqdm.trange(100):
        space = DictSpace(
            array=ArraySpace(np.array([1, 1, 1])),
            angula = Angular6dSpace((2,)),
            dict=DictSpace(
                a = ArraySpace(np.array([2,3,3]))
            )
        )
        a = space.sample()
        # numpy
        out = np.concatenate((
            a['array'].data.reshape(-1), a['angula'].data.reshape(-1), a['dict']['a'].data.reshape(-1)))

        assert out.shape[0] == space.size
        assert check_np_array(a.numpy(), out), "numpy error"

    space = DictSpace(
        array=ArraySpace(np.array([1, 1, 1])),
        angula = Angular6dSpace((2,)),
        dict=DictSpace(
            a = ArraySpace(np.array([2,3,3]))
        )
    )
    a = space.sample()
    assert isinstance(a.numpy(), np.ndarray)
    b = space.from_numpy(a.numpy())

    out = np.concatenate((
        a['array'].data.reshape(-1), a['angula'].data.reshape(-1), a['dict']['a'].data.reshape(-1)))

    assert check_np_array(b.numpy(),  out), "cpu tensor error"

    assert check_np_array(a.tensor('cpu').numpy(),  out), "cpu tensor error"
    assert check_np_array(a.tensor('cuda:0').numpy(),  out), "cuda:0 tensor error"

    # cuda
    a = a.to('cuda:0')
    assert isinstance(a['array'].data, torch.Tensor)
    assert check_np_array(a.numpy(), out) #cuda

    # operator
    #a = Array(np.array([1,2,3, 5]))
    #b = Array(np.array([2,3,4, 9]))

    a = space.sample()
    b = space.sample()
    assert np.abs((a-a).metric()) < 1e-8
    assert np.abs((a-b).metric()) > 1e-8
    #assert check_np_array((a+b).numpy(), np.array([3, 5, 7, 14]))
    #assert check_np_array((a-b).numpy(), np.array([-1, -1, -1, -4]))
    #assert np.abs(a.metric() - (1**2 + 2**2 + 3**2 + 5**2)) < 1e-5

    # batch version
    #a = Array(np.array([[1,2,3], [2,3,4]]), is_batch=True)
    #assert check_np_array(a.metric(), np.array([1**2 + 2**2 + 3**2, 2**2 + 3** 2 + 4**2]))
    c = np.stack([a.numpy(), b.numpy()])
    d = space.from_numpy(c, is_batch=True)
    assert check_np_array(d.id(0).numpy(), a.numpy())
    assert check_np_array(d.id(1).numpy(), b.numpy())
    assert check_np_array(d.numpy(), c)

    # in
    #assert [0.1, 0.1, 0.1] in space
    #assert [-0.1, -0.1, -0.1] in space
    #assert not [-1.1, -0.1, -0.1] in space


if __name__ == '__main__':
    test_array_space()
