from robot.model.arm.envs.cartpole import GoalCartpole
import numpy as np
import time


class CartpoleController:
    def __init__(self, env: GoalCartpole, p=0):
        self.env = env.unwrapped
        self.p = p

    def __call__(self, state):
        # pass
        #raise NotImplementedError
        #self.
        if np.random.random() < self.p:
            return self.env.action_space.sample()

        state_vector = self.env.state_vector()

        state, goal = state['observation'], state['desired_goal']

        qpos = state[0:2]
        qvel = state[2:4]
        achieved = np.array([state[4], 0, state[5]])
        goal = np.array((goal[0], 0, goal[1]))

        self.env.agent.set_qpos(qpos)
        self.env.agent.set_qvel(qvel)

        jac = self.env.get_jacobian()[0]

        delta = np.linalg.lstsq(jac[:3], goal-achieved)[0] * 10 # desired_velocity
        q_delta = qvel.copy() * 0
        q_delta[:] = delta

        qacc = (q_delta - qvel)/self.env.dt * 10
        #print('qacc, dt', qacc, self.env.dt, self.env.frame_skip)

        qf = self.env.agent.compute_inverse_dynamics(qacc)[self.env._actuator_dof['agent']]
        #print('QF', qf)
        self.env.load_state_vector(state_vector)

        action = qf/self.env._actuator_range['agent'][:, 1]
        #print('ACTION', action)
        return action


def main():
    env = GoalCartpole()
    policy = CartpoleController(env, p=0.0)
    from robot.utils.rl_utils import eval_policy

    eval_policy(policy, env, save_video=1, progress_episode=True, timestep=100)
    #exit(0)

    """
    obs = env.reset()
    for i in range(100):
        a = policy(obs)
        #a = env.action_space.sample()
        obs, r, done, _ = env.step(a)
        #print(done)
        """


if __name__ == '__main__':
    main()