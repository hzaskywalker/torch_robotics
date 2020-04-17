from robot import tr
import numpy as np
import torch

def check(a, b):
    assert torch.abs(a-b).sum() < 1e-6, f"{a} {b}"

def test():
    torch.manual_seed(0)
    np.random.seed(0)

    dtype=torch.float64
    bodies = []
    for i in range(10):
        R = tr.projectSO3(torch.rand((3, 3), dtype=dtype))
        p = torch.rand((3,), dtype=dtype)
        inertia = torch.rand((3, 3), dtype=dtype)
        inertia = inertia + inertia.transpose(-1, -2)
        m = torch.rand((), dtype=dtype)

        body = tr.RigidBody(tr.Rp_to_trans(R, p), inertia, m)

        bodies.append(body)

    out = bodies[0] + bodies[1:]
    out1 = bodies[0].sum_boides(bodies)
    check(out.cmass, out1.cmass)
    check(out.inertia, out1.inertia)
    check(out.mass, out1.mass)

    aligned = out.align_principle()
    assert aligned.cmass[..., [0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]].sum() < 1e-6
    check(aligned.spatial_mass_matrix(out.cmass), out.G)




if __name__ == '__main__':
    test()