"""
simply test the robot arm environment with finite differential and KL-constrained ILQG
"""
import numpy as np
from robot.controller import gps2
from robot.envs.arms.env import Env

def test_KL_iLQG():
    name = 'arm3'
    env = Env(name)
    np.random.seed(2)

    dX, dU = env.dof * 2, env.dof

    initial = gps2.LinearGaussian(
        np.zeros((dX, dX + dU)), np.zeros((dX,)), np.eye(dX) * 0.01, chol_sigma=np.zeros(dX)
    )
    target = env.gen_target()
    if name == 'arm3':
        gps2.LQR_control(env, initial, target, T=150, num_iters=20, epsilon=100000., min_epsilon_ratio=0.01, max_epsilon_ratio=100)
    else:
        gps2.LQR_control(env, initial, target, T=50, num_iters=20, epsilon=10.)


if __name__ == '__main__':
    test_KL_iLQG()
