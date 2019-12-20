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


# how to check a

if __name__ == '__main__':
    test_LQGbackward1()