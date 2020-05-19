# test the cone operators ...
import numpy as np
import torch
from robot import tr
from robot.torch_robotics.solver.cone import Orthant, SecondOrder

def check(a, b, msg=""):
    assert (a-b).abs().max() < 1e-6, f"{msg} {a} {b}"

def test_orthant():
    cone = Orthant(5)

    x = tr.togpu([
        [1, 2, 3, 4, 5],
        [-1, 2, -3, 4, 5],
        [-1, -2, -3, -4, -5]
    ])
    print(cone.inside(x))

    a = tr.togpu([
        [2, 2, 2, 2, 2],
        [1, 1, 1, 3, 1],
    ])

    b = tr.togpu([
        [1, 1, 3, 4, 5],
        [1, 1, 2, 5, 4],
    ])

    print(cone.sprod(a, b))

    check(cone.sprod(a, cone.sinv(a, b)), b, "orthant scaling")

    s, z = a, b
    W = cone.compute_scaling(s, z)
    WinvTs = cone.scale(W, s, trans=True, inverse=True)
    Wz = cone.scale(W, z)
    check(WinvTs, Wz, "orthant scaling")

    m = cone.as_matrix(W, inverse=True, trans=True)
    WinvTs2 = tr.dot(m, s)
    check(WinvTs2, WinvTs)

    m2 = cone.as_matrix(W)
    Wz2 = tr.dot(m2, z)
    check(Wz2, Wz)

    check(m, torch.inverse(tr.transpose(m2)))

    w = W['w']
    # H(w)s = z
    check(cone.scale2(w**2, s), z)
    check(cone.scale2(s, s), z *0 + 1)
    check(cone.scale2(z, z), z *0 + 1)

    check(cone.scale2(w, s), WinvTs)

    a = tr.togpu([
        [-1, 2, 3, -3, 4],
        [1, 1, 1, -3, 1],
    ])

    b = tr.togpu([
        [1, 1, 2, 2, 1],
        [3, 8, 3, 4, 1],
    ])
    assert not cone.inside(a).all()

    print(cone.max_step(a))
    assert cone.inside(cone.max_step2(b, a)[..., None] * b + a).all()


def test_second_order_cone():
    m = 12
    cone = SecondOrder(3, 4)
    a = tr.togpu([[1, 0, 0, 0, 3, 1, 1, 0, 4, 2, 1, 1],
                  [1, 0, 0, 0, 3, -2, -2, 0, 4, -2, -1, 1],
                  [1, 1, 1, 0, 3, 2, 2, 0, 4, -2, -4, 2],
                  [1, 0, 0, 0, -3, 1, 1, 0, 4, 2, 1, 1],
                  ])

    print(cone.inside(a))
    a = tr.togpu([[1, 0, 0, 0, 3, 1, 1, 0, 4, 2, 1, 1],
                  [1, 0, 0, 0, 3, -2, -2, 0, 4, -2, -1, 1]])
    b = tr.togpu([[4, 2, 2, 2, 3, -1, 1, 0, 4, 2, 1, 1],
                  [1, 0, 0, 0, 3, -2, -2, 0, 4, -2, -1, 1]])

    assert cone.inside(a).all() and cone.inside(b).all()
    from cvxopt.misc import sprod, matrix, compute_scaling
    dims = {'l':0, 'q': [4,4,4,4,4,4], 's': []}
    def make_matrix(a):
        return matrix(list(a.detach().cpu().numpy().reshape(-1)))
    a_c = make_matrix(a)
    b_c = make_matrix(b)
    sprod(a_c, b_c, dims)
    out = tr.togpu(np.array(a_c)[:, 0]).reshape(-1, m)
    check(out, cone.sprod(a, b))

    s, z = a, b
    W = cone.compute_scaling(s, z)

    a_c = make_matrix(a)
    b_c = make_matrix(b)
    lmbda = make_matrix(a)
    W2 = compute_scaling(a_c, b_c, lmbda, dims)
    check(tr.togpu(W2['beta']).reshape(-1), W['beta'].reshape(-1))
    #print(W2['v'])
    W2v = np.stack([np.array(i)[:, 0] for i in W2['v']])
    check(tr.togpu(W2v), W['v'])
    #print(np.array(lmbda).reshape(-1, 4), W['lambda'])
    check(tr.togpu(np.array(lmbda)).reshape(-1, 12), W['lambda'], 'lambda')

    WinvTs = cone.scale(W, s, trans=True, inverse=True)
    Wz = cone.scale(W, z)
    check(WinvTs, W['lambda'], "WinvTs")
    check(Wz, W['lambda'], "Wz")
    check(Wz, WinvTs)

    m = cone.as_matrix(W, inverse=True, trans=True)
    WinvTs2 = tr.dot(m, s)
    check(WinvTs2, WinvTs)

    m2 = cone.as_matrix(W)
    Wz2 = tr.dot(m2, z)
    check(Wz2, Wz)

    check(m, torch.inverse(tr.transpose(m2)))

    v = W['v'].reshape(-1, 12)
    w_ = cone.sprod(v, v)
    w = (w_.reshape(-1, 4) * W['beta']).reshape(-1, 12)
    check(cone.scale2(cone.sprod(w, w), s), z)
    check(cone.scale2(s, s), cone.identity[None, :].expand(s.shape[0], -1))
    check(cone.scale2(z, z), cone.identity[None, :].expand(s.shape[0], -1))

    a = tr.togpu([[1, 0, 0, 0, 3, 1, 1, 0, 4, 2, 1, 1],
                  [1, 0, 0, 0, 3, -2, -2, 0, 4, -2, -1, 1],
                  [1, 1, 1, 0, 3, 2, 2, 0, 4, -2, -4, 2],
                  [1, 0, 0, 0, -3, 1, 1, 0, 4, 2, 1, 1],
                  ])

    b = tr.togpu([[2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1],
                  [2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1],
                  [2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1],
                  [2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1],
                  ])

    #check(cone.scale2(w, s), WinvTs)

    cc = a + cone.max_step(a)[:, None] * cone.identity[None, :]
    assert cone.inside(cc).all()

    a_c = make_matrix(a)
    b_c = make_matrix(b)
    from cvxopt.misc import scale2
    dims = {'l':0, 'q': [4,4,4, 4,4,4, 4,4,4, 4,4,4], 's': []}
    scale2(b_c, a_c, dims)

    out = tr.togpu(np.array(a_c)).reshape(a.shape)
    check(out, cone.scale2(b, a))

    # TODO: there are numerical issue ..
    assert cone.inside((cone.max_step2(b, a)[..., None]+1e-10) * b + a).all()


if __name__ == '__main__':
    #test_orthant()
    test_second_order_cone()
