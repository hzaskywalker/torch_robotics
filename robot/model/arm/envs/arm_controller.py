# this controller is used to sample the best policy
import torch
import numpy as np
from robot.envs.hyrule.rl_env import ArmReachWithXYZ


class Controller:
    def __init__(self, env: ArmReachWithXYZ, p=0):
        self.env = env
        self.p = p

    def __call__(self, state):
        # pass
        #raise NotImplementedError
        #self.
        if np.random.random() < self.p:
            return self.env.action_space.sample()

        state_vector = self.env.state_vector()
        state, goal = state['observation'], state['desired_goal']

        qpos = state[:7]
        qvel = state[7:14]
        achieved = state[14:17]

        self.env.agent.set_qpos(qpos)
        self.env.agent.set_qvel(qvel)

        jac = self.env.get_jacobian()

        q_delta = np.linalg.lstsq(jac[:3], goal-achieved)[0] * 3 # desired_velocity
        qacc = (q_delta - qvel)/self.env.dt

        qf = self.env.agent.compute_inverse_dynamics(qacc) + self.env.agent.compute_passive_force()
        self.env.load_state_vector(state_vector)
        return qf/50  # again, private attribute...

class RandomController:
    def __init__(self, env: ArmReachWithXYZ):
        self.action_space = env.action_space

    def __call__(self, obs):
        return self.action_space.sample()


if __name__ == '__main__':
    env = ArmReachWithXYZ()
    policy = Controller(env, p=0.0)
    from robot.utils.rl_utils import eval_policy

    eval_policy(policy, env, save_video=0, progress_episode=True, timestep=100, eval_episodes=100)
