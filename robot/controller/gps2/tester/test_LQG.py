import numpy as np
import tqdm
from robot.controller import gps2
from typing import List

def test_LQGbackward1():
    W = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ]
    )
    b = np.array(
        [0, 0]
    )

    dX = 2
    dU = 2

    T = 2
    dynamics = []
    l_xx = []
    l_x = []
    for i in range(T):
        dynamics.append(gps2.LinearGaussian(W, b, 0))
        l_xx_t, l_x_t = gps2.cost_fk(dX, dU, [0, 0], 'u')
        if i == T-1:
            # add for final
            l_xx_goal, l_x_goal = gps2.cost_fk(dX, dU, [1, 1], 'x')
            l_xx_t += l_xx_goal * 1000
            l_x_t += l_x_goal * 1000
        l_xx.append(l_xx_t)
        l_x.append(l_x_t)
    print(l_x[-1])

    traj = gps2.LQGbackward(dynamics, l_xx, l_x)
    print(traj[0])
    print(traj[1])


def generate_dynamics():
    T = 10
    A = np.random.normal(size=(11, 4, 7))
    B = np.random.normal(size=(11, 4,))
    C = np.random.normal(size=(11, 4, 4)) * 0.1
    dynamics = []
    l_xuxu = []
    l_xu = []
    initial = gps2.LinearGaussian(A[-1], B[-1], C[-1].T.dot(C[-1]))
    for i in range(T):
        dynamics.append(
            gps2.LinearGaussian(A[i], B[i], C[i].T.dot(C[i]))  # dynamics
        )
        sig = np.random.normal(size=(7, 7))
        l_xuxu.append(sig.T.dot(sig))
        l_xu.append(np.random.normal(size=7))
    return dynamics, initial, l_xuxu, l_xu


def test_LQG2():
    """
    Test LQG under deterministic policy
    """
    for i in range(1000):
        dynamics, initial, l_xuxu, l_xu = generate_dynamics()
        T = len(dynamics)
        traj = gps2.LQGbackward(dynamics, l_xuxu, l_xu)
        a = gps2.LQGeval(traj, dynamics, initial, l_xuxu, l_xu)
        step = np.random.choice([0.1, 0.01, 0.001])
        for i in range(T):
            traj[i].W += np.random.normal(size=traj[i].W.shape) * step
            traj[i].b += np.random.normal(size=traj[i].b.shape) * step
        b = gps2.LQGeval(traj, dynamics, initial, l_xuxu, l_xu)
        assert a < b, f"{a}, {b}"


def test_soft_kl():
    for i in tqdm.trange(1000):
        eta = 1
        dynamics, initial, l_xuxu, l_xu = generate_dynamics()
        dX, dU = initial.dY, dynamics[0].dX - dynamics[0].dY
        T = len(dynamics)
        prev_traj: List[gps2.LinearGaussian] = gps2.LQGbackward(dynamics, l_xuxu, l_xu)

        """
        Qtt, Qt = prev_traj[0].log_derivative()
        idx_x = slice(prev_traj[0].dX)
        idx_u = slice(prev_traj[0].dX, prev_traj[0].dX + prev_traj[0].dY)
        print(-np.linalg.pinv(Qtt[idx_u, idx_u]).dot(Qtt[idx_u, idx_x]))
        print(prev_traj[0].W)
        exit(0)
        """

        #prev_traj = [
        #    gps2.LinearGaussian(np.zeros((dU, dX)), np.zeros((dU,)), np.eye(dU)) for i in range(T)
        #]
        #print('prev_traj score', gps2.LQGeval(prev_traj, dynamics, initial, l_xuxu, l_xu, entropy=True))
        prev_traj_score = gps2.LQGeval(prev_traj, dynamics, initial, l_xuxu, l_xu, entropy=True)
        traj, _eta = gps2.soft_KL_LQG(dynamics, l_xuxu, l_xu, prev_traj=prev_traj, eta=eta, delta=1e-4)
        kl_div = gps2.kl_divergence(traj, prev_traj, dynamics, initial)
        a = gps2.LQGeval(traj, dynamics, initial, l_xuxu, l_xu, entropy=False)

        traj2, _eta = gps2.soft_KL_LQG(dynamics, l_xuxu, l_xu, prev_traj=prev_traj, eta=1000, delta=1e-4)
        #print('kl', kl_div)
        #print('now score', a)
        #print('final score', final)
        final = a+ eta * kl_div
        final2 = gps2.LQGeval(traj2, dynamics, initial, l_xuxu, l_xu, entropy=False) +\
                 1 *gps2.kl_divergence(traj2, prev_traj, dynamics, initial)
        #print(prev_traj_score, final, final2)
        #assert prev_traj_score < final
        #exit(0)
        assert final < final2

if __name__ == '__main__':
    test_soft_kl()