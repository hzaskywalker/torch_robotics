import torch
from robot.envs.spaces.graph import *

def check_np_array(a, b):
    return ((a-b)**2).sum() < 1e-6


def test():
    n = 5
    m = 8
    space = graph_space(node_space(n), edge_space(n, m, 1), None)

    a = space.sample()
    assert isinstance(a.numpy(), np.ndarray)
    b = space.from_numpy(a.numpy())

    out = np.concatenate((
        a['node']['p'].data.reshape(-1), a['node']['v'].data.reshape(-1), a['node']['w'].data.reshape(-1), a['node']['dw'].data.reshape(-1), a['edge']['e'].data.reshape(-1), a['edge']['s'].data.reshape(-1), a['edge']['t'].data.reshape(-1)))

    assert check_np_array(b.numpy(),  out), "cpu tensor error"

    assert check_np_array(a.tensor('cpu').numpy(),  out), "cpu tensor error"
    assert check_np_array(a.tensor('cuda:0').numpy(),  out), "cuda:0 tensor error"

    # cuda
    a = a.to('cuda:0')
    assert isinstance(a['node']['p'].data, torch.Tensor)
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
    test()