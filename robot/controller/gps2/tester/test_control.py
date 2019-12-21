"""
simply test the robot arm environment with finite differential and KL-constrained ILQG
"""
import numpy as np
from robot.controller import gps2
from robot.envs.arms.env import Env

def test_KL_ILQG():
    env = Env('arm2')
    np.random.seed(0)

    dX, dU = env.dof * 2, env.dof

    initial = gps2.LinearGaussian(
        np.zeros((dX, dX + dU)), np.zeros((dX,)), np.eye(dX) * 0.01
    )
    target = env.gen_target()
    gps2.LQR_control(env, initial, target, T=50, num_iters=100)


if __name__ == '__main__':
    test_KL_ILQG()
