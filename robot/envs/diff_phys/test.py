import gym
import torch
import tqdm
import numpy as np
import cv2
from robot.envs.diff_phys.acrobat import GoalAcrobat

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


def test_articulation():
    env = GoalAcrobat()
    articulator = env.articulator
    print(articulator.forward_kinematics(torch.tensor([np.pi/2, np.pi/2], dtype=torch.float32, device='cuda:0')))
    qpos = torch.tensor([-np.pi/2, np.pi/2], dtype=torch.float32, device='cuda:0')
    env.articulator.set_qpos(qpos)
    img = env.render(mode='rgb_array')
    cv2.imwrite('x.jpg', img)

    print(env.observation_space)
    print(env.action_space)

    #obs = env.reset()
    #for i in tqdm.trange(10000):
    #    a = env.action_space.sample()
    #    env.step(a)

    def write():
        for i in tqdm.trange(60):
            a = env.action_space.sample()
            img = env.render(mode='rgb_array')
            yield img
            env.step(a)
    from robot.utils import write_video
    write_video(write())



if __name__ == '__main__':
    #test_acrobat()
    test_articulation()
