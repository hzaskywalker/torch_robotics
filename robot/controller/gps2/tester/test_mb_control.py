"""
simply test the robot arm environment with finite differential and KL-constrained ILQG
"""
import numpy as np
from robot.controller import gps2
from robot.envs.arms.env import Env
from robot.model.gmm_linear_model import GaussianLinearModel

def test_KL_iLQG():
    name = 'arm3'
    env = Env(name)
    np.random.seed(3)

    dX, dU = env.dof * 2, env.dof

    initial = gps2.LinearGaussian(
        np.zeros((dX, dX + dU)), np.zeros((dX,)), np.eye(dX) * 0.1, chol_sigma=np.zeros(dX)
    )
    target = env.gen_target()

    T = 100 if name == 'arm2' else 150
    epsilon = 100 if name == 'arm2' else 20 #200.
    samples = 4 if name == 'arm2' else 20 #8
    initial_std = 2 if name == 'arm2' else 2.

    # many samples, and less priors can make the model work?
    model = GaussianLinearModel(regularization=1e-6, prior_strength=1, min_samples_per_cluster=40,
                                max_clusters=50, max_samples=5 * samples * T)

    solver = gps2.ILQGTrajOpt(env, initial, target, T, samples, model, epsilon=epsilon, use_model=False, initial_std=initial_std)
    solver.run(200)


if __name__ == '__main__':
    test_KL_iLQG()
