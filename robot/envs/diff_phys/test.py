import gym
import tqdm
import torch
import tqdm
import numpy as np
import cv2
from robot.envs.diff_phys.acrobat import GoalAcrobat

from robot import torch_robotics as tr
import robot.utils as tru


def test_acrobat():
    from robot.envs.diff_phys.acrobat_gt import AcrobotEnv
    env = AcrobotEnv()
    #env = gym.make('Acrobot-v1')
    env.reset()
    env.state = np.zeros((2,))
    img = env.render(mode='rgb_array')
    cv2.imwrite('x.jpg', img)
    exit(0)
    #cv2.waitKey(0)
    for i in tqdm.trange(10000):
        action = env.action_space.sample()
        img = env.render()
        env.step(action)


def solveIK(env):
    from sklearn.linear_model import LinearRegression
    tmp = env._get_obs()
    state, goal = tmp['observation'], tmp['desired_goal']

    dim = (state.shape[0] - 2) // 2
    qpos = state[:dim]
    qvel = state[dim:dim * 2]

    model = env.articulator

    for i in range(200):
        env.set_state(qpos, qvel * 0)
        img = env.render(mode='rgb_array')
        achieved = env._get_obs()['achieved_goal']
        jac = env.get_jacobian()  # notice that the whole system is (w, v)

        delta = np.dot(np.linalg.pinv(jac[:2]), (goal-achieved)[:2])
        qpos += delta * env.dt
        #print(delta * env.dt)

        cv2.imshow('x.jpg', img)
        cv2.waitKey(0)
    return qpos


def test_articulation():
    env = GoalAcrobat()
    articulator = env.articulator
    print(articulator.forward_kinematics(torch.tensor([np.pi/2, np.pi/2], dtype=torch.float64, device='cuda:0')))
    qpos = torch.tensor([-np.pi/2, np.pi/2], dtype=torch.float64, device='cuda:0')
    env.articulator.set_qpos(qpos)
    img = env.render(mode='rgb_array')
    cv2.imwrite('x.jpg', img)

    print(env.observation_space)
    print(env.action_space)
    """
    env.reset()
    qpos = solveIK(env)
    env.articulator.set_qpos(qpos)
    exit(0)
    """


    """
    state = env.state_vector()
    obs1 = env._get_obs()['observation']
    env.step(np.array([1, 2]))

    env.load_state_vector(state)
    obs2 = env._get_obs()['observation']
    """
    from robot.envs.diff_phys.acrobat import IKController
    policy = IKController(env)
    tru.eval_policy(policy, env, save_video=1, eval_episodes=50, progress_episode=True, timestep=50)
    exit(0)

    def write():
        for i in tqdm.trange(60):
            a = env.action_space.sample()
            img = env.render(mode='rgb_array')
            yield img
            obs, r, done, _ = env.step(a)
    tru.write_video(write())


def test_batched_env():
    # pass
    from robot.envs.diff_phys.acrobat import IKController

    env = GoalAcrobat(batch_size=2)
    obs = env.reset()
    controller = IKController(env)
    num = 10

    cc = 0
    for j in range(num):
        idx = 0
        obs = env.reset()
        for i in tqdm.trange(50):
            a = controller(obs)
            obs, r, done, info = env.step(a)
            #img = env.render(mode='rgb_array')
            #cv2.imshow('x', img)
            #cv2.waitKey(0)
        cc += info['is_success'].mean()
    print(cc/num)


if __name__ == '__main__':
    #test_acrobat()
    #test_articulation()
    test_batched_env()
