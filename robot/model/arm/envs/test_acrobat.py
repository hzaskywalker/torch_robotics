from robot.model.arm.envs.acrobat import GoalAcrobat
import numpy as np
import time
from sklearn.linear_model import Ridge


class AcrobatController:
    def __init__(self, env: GoalAcrobat, p=0):
        self.env = env.unwrapped
        self.p = p
        self.clf = Ridge(alpha=0.01)

    def __call__(self, state):
        # pass
        #raise NotImplementedError
        #self.
        if np.random.random() < self.p:
            return self.env.action_space.sample()

        state_vector = self.env.state_vector()

        state, goal = state['observation'], state['desired_goal']

        dim = (state.shape[0] - 2)//3
        qpos = state[:dim]
        qvel = state[dim:dim*2]

        achieved = np.array([state[-2], state[-1]])
        goal = np.array((goal[0], goal[1]))

        self.env.agent.set_qpos(qpos)
        self.env.agent.set_qvel(qvel)

        jac = self.env.get_jacobian()[0]

        #delta = np.linalg.lstsq(jac[:3], goal-achieved)[0] # desired_velocity
        diff = np.array((goal[0]-achieved[0], 0, goal[1] - achieved[1]))
        self.clf.fit(jac[:3], diff)
        delta = self.clf.coef_ * 10

        q_delta = qvel.copy() * 0
        q_delta[:] = delta

        qacc = (q_delta - qvel)/self.env.dt
        #print('qacc, dt', qacc, self.env.dt, self.env.frame_skip)

        #print(delta, qvel, self.env._actuator_dof['agent'], self.env.agent.compute_inverse_dynamics(qacc).shape)
        dofs = self.env._actuator_dof['agent']
        qf = self.env.agent.compute_inverse_dynamics(qacc)[dofs]
        #print('QF', qf)
        self.env.load_state_vector(state_vector)
        qf += self.env.agent.compute_passive_force()
        action = qf/self.env._actuator_range['agent'][:, 1]
        #print('ACTION', action)
        return action


def main():
    env = GoalAcrobat(length=[0.3, 0.3, 0.3])
    policy = AcrobatController(env, p=0.0)

    from robot.utils.rl_utils import eval_policy
    eval_policy(policy, env, save_video=1, eval_episodes=100, progress_episode=True, timestep=50)
    exit(0)

    obs = env.reset()
    for i in range(1000):
        #a = env.action_space.sample()
        env.render()
        a = policy(obs)
        obs, r, done, _ = env.step(a)


if __name__ == '__main__':
    main()