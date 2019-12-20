import numpy as np
from robot.controller import gps2

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


def test_LQG2():
    """
    Test LQG under deterministic policy
    """
    for i in range(1000):
        T = 10
        A = np.random.normal(size=(11, 4, 7))
        B = np.random.normal(size=(11, 4,))
        dynamics = []
        l_xuxu = []
        l_xu = []
        initial = gps2.LinearGaussian(A[-1], B[-1])
        for i in range(T):
            dynamics.append(
                gps2.LinearGaussian(A[i], B[i]) #dynamics
            )
            sig = np.random.normal(size=(7, 7))
            l_xuxu.append(sig.T.dot(sig))
            l_xu.append(np.random.normal(size=7))

        traj = gps2.LQGbackward(dynamics, l_xuxu, l_xu)
        for i in traj:
            # set it to deterministic
            i.sigma = i.sigma * 0

        a = gps2.LQGeval(traj, dynamics, initial, l_xuxu, l_xu)
        for i in range(T):
            traj[i].W += np.random.normal(size=traj[i].W.shape) * 0.01
            traj[i].b += np.random.normal(size=traj[i].b.shape) * 0.01
        b = gps2.LQGeval(traj, dynamics, initial, l_xuxu, l_xu)
        assert a < b, f"{a}, {b}"

if __name__ == '__main__':
    test_LQG2()