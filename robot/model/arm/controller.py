# this controller is used to sample the best policy
import numpy as np
from robot.envs.hyrule.rl_env import ArmReachWithXYZ


class Controller:
    def __init__(self, env: ArmReachWithXYZ):
        self.env = env

    def __call__(self, state):
        # pass
        #raise NotImplementedError
        #self.
        state_vector = self.env.state_vector()

        state, goal = state['observation'], state['desired_goal']

        assert len(state) == 29
        qpos = state[:13]
        qvel = state[13:26]
        achieved = state[26:29]

        self.env.agent.set_qpos(qpos)
        self.env.agent.set_qvel(qvel)

        jac = self.env.get_jacobian()[0]

        pos_ctrl = (goal - achieved)/0.3
        delta = np.linalg.lstsq(jac[:3], pos_ctrl)[0] # desired_velocity
        q_delta = qvel.copy() * 0
        q_delta[self.env._actuator_dof['agent']] = delta

        qf = self.env.agent.compute_drive_force(q_delta - qvel)[self.env._actuator_dof['agent']]
        self.env.load_state_vector(state_vector)
        return qf/self.env._actuator_range['agent'][:, 1]


if __name__ == '__main__':
    env = ArmReachWithXYZ()
    policy = Controller(env)
    from robot.envs.sapien.exp.utils import eval_policy

    eval_policy(policy, env, save_video=1, progress_episode=True, timestep=50)
