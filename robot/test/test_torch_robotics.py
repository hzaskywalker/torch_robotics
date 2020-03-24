import torch
import time
import numpy as np
import robot.torch_robotics as tr
import robot.modern_robotics as mr

def togpu(a):
    if not isinstance(a, torch.Tensor):
        return torch.tensor(a, dtype=torch.float64, device='cuda:0')
    else:
        return a.cuda()

def check(a, b):
    assert torch.abs(a-b).sum() < 1e-6, f"{a} {b}"

def test_inverse_dynamics():
    print('test inverse dynamics')
    thetalist = np.array([0.1, 0.1, 0.1])
    dthetalist = np.array([0.1, 0.2, 0.3])
    ddthetalist = np.array([2, 1.5, 1])
    taulist = np.array([0.5, 0.6, 0.7])
    M01 = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0.089159],
                    [0, 0, 0, 1]])
    M12 = np.array([[0, 0, 1, 0.28],
                    [0, 1, 0, 0.13585],
                    [-1, 0, 0, 0],
                    [0, 0, 0, 1]])
    M23 = np.array([[1, 0, 0, 0],
                    [0, 1, 0, -0.1197],
                    [0, 0, 1, 0.395],
                    [0, 0, 0, 1]])
    M34 = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0.14225],
                    [0, 0, 0, 1]])
    G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
    G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
    G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
    Glist = np.array([G1, G2, G3])
    Mlist = np.array([M01, M12, M23, M34])
    Slist = np.array([[1, 0, 1, 0, 1, 0],
                      [0, 1, 0, -0.089, 0, 0],
                      [0, 1, 0, -0.089, 0, 0.425]]).T

    g = np.array([0, 0, -9.8])
    Ftip = np.array([1, 1, 1, 1, 1, 1])
    gt = togpu(mr.InverseDynamics(thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, Glist, Slist))
    mass = togpu(mr.MassMatrix(thetalist, Mlist, Glist, Slist))[None, :]
    c_theta_dtheta = togpu(mr.VelQuadraticForces(thetalist, dthetalist, Mlist, Glist, Slist))[None, :]
    passive_gravity = togpu(mr.GravityForces(thetalist, g, Mlist, Glist, Slist))[None, :]
    passive_ftip = togpu(mr.EndEffectorForces(thetalist, Ftip, Mlist, Glist, Slist))[None, :]

    forward = togpu(mr.ForwardDynamics(thetalist, dthetalist, taulist, g, Ftip, Mlist, Glist, Slist))[None, :]
    Mi = np.eye(4)
    for i in range(len(Mlist)):
        Mi = np.dot(Mi,Mlist[i])
    ee_trans = togpu(mr.FKinSpace(Mi, Slist, thetalist))[None,:]

    jacobian = togpu(mr.JacobianSpace(Slist, thetalist))[None,:]

    theta = togpu(thetalist)[None, :]
    dtheta = togpu(dthetalist)[None, :]
    ddtheta = togpu(ddthetalist)[None, :]

    def make_tensor(A):
        return torch.stack([togpu(i) for i in A], dim=0)[None, :]
    G = make_tensor(Glist)
    M = make_tensor(Mlist)
    S = make_tensor(Slist.T)

    gravity = togpu(g)[None, :]
    ftip = togpu(Ftip)[None,:]
    begin = time.time()

    A = tr.S_to_A(S, M)
    output = tr.inverse_dynamics(theta, dtheta, ddtheta, gravity, ftip, M, G, A)
    check(output[0], gt)
    print('passed', time.time() - begin)

    print('test mass matrix')
    mass2 = tr.compute_mass_matrix(theta, M, G, A)
    check(mass2, mass)
    print('passed')

    print('test passive')
    g, f = tr.compute_passive_force(theta, M, G, A, gravity, ftip)
    check(g, passive_gravity)
    check(f, passive_ftip)
    print('passed')

    print('test coriolis')
    output2 = tr.compute_coriolis_centripetal(theta, dtheta, M, G, A)
    check(output2, c_theta_dtheta)
    print('passed')

    print('test forward dynamics')
    tau = togpu(taulist)[None, :]
    forward2 = tr.forward_dynamics(theta, dtheta, tau, gravity, ftip, M, G, A)
    check(forward2, forward)
    print('passed')

    print('test forward kinematics')
    Ts = tr.fk_in_space(theta, M, A)
    check(Ts[:, -1], ee_trans)
    print('passed')

    print('test jacobian')
    Jac = tr.jacobian_space(theta, M, A)
    check(Jac, jacobian)
    print('passed')


def test_vec_to_se3():
    print("test vec to se3")
    V = np.array([1, 2, 3, 4, 5, 6])
    check(tr.vec_to_se3(togpu(V)[None,:])[0], togpu(mr.VecTose3(V)))
    print("passed")

def test_inv_trans():
    print("test inv_trans")
    T = np.array([[1, 0, 0, 0],
                  [0, 0, -1, 0],
                  [0, 1, 0, 3],
                  [0, 0, 0, 1]])
    check(tr.inv_trans(togpu(T)[None,:])[0], togpu(mr.TransInv(T)))
    print("passed")


def test_adjoint():
    print("test adjoint")
    T = np.array([[1, 0, 0, 0],
                  [0, 0, -1, 0],
                  [0, 1, 0, 3],
                  [0, 0, 0, 1]])
    check(tr.Adjoint(togpu(T)[None,:])[0], togpu(mr.Adjoint(T)))
    print("passed")


def test_expso3():
    print("test expso3")
    so3mat = np.array([[0, -3, 2],
                       [3, 0, -1],
                       [-2, 1, 0]])
    check(tr.expso3(togpu(so3mat)[None, :])[0], togpu(mr.MatrixExp3(so3mat)))

    so3mat = np.array([[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]])
    check(tr.expso3(togpu(so3mat)[None, :])[0], togpu(mr.MatrixExp3(so3mat)))
    print("passed")

def test_expse3():
    print("test expse3")
    se3mat = np.array([[0, -3, 2, 4],
                       [3, 0, -1, 5],
                       [-2, 1, 0, 6],
                       [0, 0, 0, 0]])
    check(tr.expse3(togpu(se3mat)[None, :])[0], togpu(mr.MatrixExp6(se3mat)))

    se3mat = np.array([[0, 0, 0, 4],
                       [0, 0, 0, 5],
                       [0, 0, 0, 6],
                       [0, 0, 0, 0]])
    check(tr.expse3(togpu(se3mat)[None, :])[0], togpu(mr.MatrixExp6(se3mat)))

    """
    so3mat = np.array([[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]])
    check(tr.expso3(togpu(so3mat)[None, :])[0], togpu(mr.MatrixExp3(so3mat)))
    """
    print("passed")


def test_logSO3():
    print("test logSO3")
    np.array([[0, 0, 1],
              [1, 0, 0],
              [0, 1, 0]]),
    SO3List = [
        np.array([[10, 0, 1],
                  [1, 10, 0],
                  [0, 1, 10]]),
        np.array([[0.2, 0, 1],
                  [1, 0.2, 0],
                  [0, 1, 0.6]]),
        np.array([[-1, 0, 2],
                  [-1, 0, 0],
                  [0, 3, 0]]),
        np.array([[0, 0, 2],
                  [-1, -1, 0],
                  [0, 3, 0.]]),
        np.array([[0, 0, 2],
                  [-1, 0, 0],
                  [0, 3, -1]]),
    ]
    check(tr.logSO3(togpu(SO3List)), togpu([mr.MatrixLog3(i) for i in SO3List]))
    print("passed")


def test_logSE3():
    print("test logSE3")

    SE3List = [
        np.array([[1, 1, 0, 1],
                  [0, 1, 1, 0],
                  [0, 0, 1, 3],
                  [0, 0, 0, 1]]),
        np.array([[1, 0, 0, 0],
                  [0, 0, -1, 0],
                  [0, 1, 0, 3],
                  [0, 0, 0, 1]]),
        np.array([[1, 0, 0, 2],
                  [0, 0, -1, 3],
                  [0, 1, 0, 3],
                  [0, 0, 0, 1]]),
        np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 3],
                  [0, 0, 0, 1]]),
        np.array([[-1, 2, 0, 1],
                  [2, -1, 0, 0],
                  [0, 0, 1, 3],
                  [0, 0, 0, 1]]),
    ]

    check(tr.logSE3(togpu(SE3List)), togpu([mr.MatrixLog6(i) for i in SE3List]))
    print("passed")


if __name__ == '__main__':
    test_logSE3()
    test_logSO3()
    test_inv_trans()
    test_adjoint()
    test_expse3()
    test_vec_to_se3()
    test_inverse_dynamics()
