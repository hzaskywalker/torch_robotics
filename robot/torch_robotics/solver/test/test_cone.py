# test the cone operators ...
import torch
from robot import tr
from robot.torch_robotics.solver.cone import Orthant

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

if __name__ == '__main__':
    test_orthant()

