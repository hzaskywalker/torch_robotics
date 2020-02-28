import gym
import numpy as np
import tqdm
from gym.wrappers import TimeLimit
import cv2

class EnvWrapper(gym.Wrapper):
    def __init__(self, env):
        env.reset()
        env.render_target = True
        env = TimeLimit(env, 100)
        super(EnvWrapper, self).__init__(env)

    def reset(self, **kwargs):
        self.env.reset()
        #self.unwrapped._position = np.array([-2., -3.])
        #self.unwrapped._position = np.array([0., -3.])
        self.unwrapped._position = np.array([0., 0.])
        self.unwrapped._target_position = np.array([0, 3.5])

    def render(self, mode='human'):
        self.env.env.render()
        if mode == 'human':
            pass
        else:
            return self.env.env.get_image()

    def state_vector(self):
        return np.concatenate(([self.env._elapsed_steps], self.env.env._position, self.env.env._target_position))

    def set_state(self, a, b):
        self.env._elapsed_steps = int(a[0])
        self.env.env._position = a[1:].copy()
        self.env.env._target_position = b.copy()

    def state2obs(self, s):
        return s[1:3]


def make(env_name):
    import cv2
    import tqdm
    from multiworld.envs import pygame
    from multiworld.envs import mujoco

    if env_name == 'pm':
        env_name = 'PointmassUWallTrainEnvBig-v0'
    elif env_name == 'pnr':
        env_name = 'Image84SawyerPushAndReachTrainEnvBig-v0'
    else:
        raise NotImplementedError

    env = EnvWrapper(gym.make(env_name))
    return env


def test():
    from robot.envs.sapien.exp.utils import set_state
    env = make('pm')
    print(env.action_space.low, env.action_space.high)
    env.reset()

    state = env.state_vector()

    for i in tqdm.trange(100000):
        a = env.action_space.sample()
        img = env.render(mode='rgb_array')
        cv2.imshow('img', img)
        cv2.waitKey(0)
        o, r, done, info = env.step(a)
        print(r)

        if done:
            set_state(env, state)


if __name__ == '__main__':
    test()
