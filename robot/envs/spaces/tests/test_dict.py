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
        assert check_np_array(np.array(a), out), "numpy error"

    space = DictSpace(
        array=ArraySpace(np.array([1, 1, 1])),
        angula = Angular6dSpace((2,)),
        dict=DictSpace(
            a = ArraySpace(np.array([2,3,3]))
        )
    )
    a = space.sample()

    a['xx'] = 2
    assert 'xx' not in a.__dict__
    assert 'xx' in a
    assert a.xx == 2

    a.yy = 3
    assert a.yy == 3
    assert 'yy' not in a
    assert 'yy' in a.__dict__

    a = space.sample()

    assert isinstance(np.array(a), np.ndarray)

    a.array.data = np.array([1, 2, 3])
    out = np.concatenate((
        np.array([1, 2, 3]), a['angula'].data.reshape(-1), a['dict']['a'].data.reshape(-1)))

    b = space.from_numpy(a.numpy().serialize())

    assert check_np_array(np.array(b),  out), "cpu tensor error"

    assert check_np_array(np.array(a.tensor('cpu')),  out), "cpu tensor error"
    assert check_np_array(np.array(a.tensor('cuda:0')),  out), "cuda:0 tensor error"

    # cuda
    a = a.to('cuda:0')
    assert isinstance(a['array'].data, torch.Tensor)
    assert check_np_array(np.array(a), out) #cuda

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
    assert check_np_array(d.id(0).numpy().serialize(), a.numpy().serialize())
    assert check_np_array(d.id(1).numpy().serialize(), b.numpy().serialize())
    assert check_np_array(d.numpy().serialize(), c)

    assert space.contains(a)

    # in
    #assert [0.1, 0.1, 0.1] in space
    #assert [-0.1, -0.1, -0.1] in space
    #assert not [-1.1, -0.1, -0.1] in space


if __name__ == '__main__':
    test_array_space()
