from robot.model.arm.envs.cartpole import GoalCartpole
import numpy as np
import time
from sklearn.linear_model import Ridge


class CartpoleController:
    def __init__(self, env: GoalCartpole, p=0):
        self.env = env.unwrapped
        self.p = p
        self.clf = Ridge(alpha=1.)

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

        self.env.agent.set_qpos(qpos)
        self.env.agent.set_qvel(qvel)

        jac = self.env.get_jacobian()[0]

        achieved = self.env.ee_link.pose.p
        goal = np.array((goal[0], 0, goal[1]))

        #x2z = np.array([0.7071068, 0, 0.7071068, 0])
        from transforms3d import quaternions
        diff_angle = quaternions.qmult((1, 0, 0, 0), quaternions.qinverse(self.env.ee_link.pose.q))

        #print(quaternions.qmult(diff_angle, self.env.ee_link.pose.q))
        #print(self.env.ee_link.pose.q, quaternions.quat2axangle(self.env.ee_link.pose.q))
        #exit(0)
        #print(quaternions.quat2axangle(x2z))
        #print(diff_angle)
        vec, theta = quaternions.quat2axangle(diff_angle)

        diff = np.concatenate((goal-achieved, -vec * theta))
        jac = jac.astype(np.float64)
        diff = diff.astype(np.float64)

        self.clf.fit(jac[:3], goal-achieved)
        delta = self.clf.coef_ * 100

        q_delta = qvel.copy()
        q_delta[:] = delta * 10
        q_delta[0] *= 1e-2

        qacc = (q_delta - qvel)/self.env.dt
        #print('qacc, dt', qacc, self.env.dt, self.env.frame_skip)

        qf = self.env.agent.compute_inverse_dynamics(qacc)[self.env._actuator_dof['agent']]
        #qf += self.env.agent.compute_passive_force()[self.env._actuator_dof['agent']]
        #print('QF', qf)
        self.env.load_state_vector(state_vector)

        action = qf/self.env._actuator_range['agent'][:, 1]
        #print('ACTION', action)
        return action


def main():
    env = GoalCartpole()
    policy = CartpoleController(env, p=0.0)
    from robot.utils.rl_utils import eval_policy

    eval_policy(policy, env, save_video=1, eval_episodes=1, progress_episode=True, timestep=100)
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