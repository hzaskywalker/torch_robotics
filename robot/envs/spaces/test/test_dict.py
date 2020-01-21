import numpy as np
import tqdm
import torch
from robot.envs.spaces.dict import  Dict
from robot.envs.spaces.array import Array
#from robot.envs.spaces.angular import Angular6d, Angular6dSpace

def check_np_array(a, b):
    return ((a-b)**2).sum() < 1e-6

def test_array_space():
    for i in tqdm.trange(100):
        space = Dict(
            array=Array(np.array([1, 1, 1])),
            angula = Array(np.array([2,])),
            dict=Dict(
                a = Array(np.array([2,3,3]))
            )
        )
        a = space(space.sample(), is_batch=False)
        # numpy
        out = np.concatenate((
            a['array'].reshape(-1), a['angula'].reshape(-1), a['dict']['a'].reshape(-1)))

        assert out.shape[0] == space.size
        assert check_np_array(np.array(a), out), "numpy error"

    space = Dict(
        array=Array(np.array([1, 1, 1])),
        angula = Array(np.array((2,))),
        dict=Dict(
            a = Array(np.array([2,3,3]))
        )
    )
    a = space(space.sample(), is_batch=False)

    assert isinstance(np.array(a), np.ndarray)

    a.array = np.array([1, 2, 3])
    out = np.concatenate((
        np.array([1, 2, 3]), a['angula'].reshape(-1), a['dict']['a'].reshape(-1)))
    assert check_np_array(np.array(a),  out), "cpu tensor error"

    b = space(space.deserialize(np.array(a)), is_batch=False)

    assert check_np_array(np.array(b),  out), "cpu tensor error"

    assert check_np_array(np.array(a.tensor('cpu')),  out), "cpu tensor error"
    assert check_np_array(np.array(a.tensor('cuda:0')),  out), "cuda:0 tensor error"

    # cuda
    a = a.tensor('cuda:0')
    assert isinstance(a.array, torch.Tensor)
    assert check_np_array(np.array(a), out) #cuda

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
    c = np.stack([np.array(a), np.array(b)])

    d = space(space.deserialize(c, is_batch=True), is_batch=True)
    assert check_np_array(d.id(0).__array__(), a.__array__())
    assert check_np_array(d.id(1).__array__(), b.__array__())
    assert check_np_array(d.__array__(), c)

    assert space.contains(a.state)

    # in
    #assert [0.1, 0.1, 0.1] in spaces
    #assert [-0.1, -0.1, -0.1] in spaces
    #assert not [-1.1, -0.1, -0.1] in spaces


if __name__ == '__main__':
    test_array_space()
