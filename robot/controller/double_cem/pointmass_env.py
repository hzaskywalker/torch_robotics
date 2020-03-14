import gym
import numpy as np
import tqdm
from gym.wrappers import TimeLimit
import cv2
from multiworld import register_all_envs
register_all_envs()

class EnvWrapper(gym.Wrapper):
    def __init__(self, env, mode=0):
        env.reset()
        env.render_target = True
        env = TimeLimit(env, 100)
        self.mode = mode
        super(EnvWrapper, self).__init__(env)

    def reset(self, **kwargs):
        self.env.reset()
        #self.unwrapped._position = np.array([-2., -3.])
        self.unwrapped._position = np.array([0., 0.])
        #self.unwrapped._target_position = np.array([-3.5, 3.5])
        self.unwrapped._target_position = np.array([0, 3.5])

    def render(self, mode='human'):
        if self.mode == 0:
            self.env.env.render()
            if mode == 'human':
                pass
            else:
                return self.env.env.get_image()
        else:
            return self.env.env.render(mode)

    def state_vector(self):
        return np.concatenate(([self.env._elapsed_steps], self.env.env._position, self.env.env._target_position))

    def set_state(self, a, b):
        self.env._elapsed_steps = int(a[0])
        self.env.env._position = a[1:].copy()
        self.env.env._target_position = b.copy()

    def state2obs(self, s):
        return s[1:3]


def make(env_name):
    mode = 0
    if env_name == 'pm':
        env_name = 'PointmassUWallTrainEnvBig-v0'
        env = EnvWrapper(gym.make(env_name), mode)
        return env
    elif env_name == 'pnr':
        mode = 1
        env_name = 'SawyerPushAndReachArenaTrainEnvBig-v0'
    elif env_name == 'pnr_torque':
        mode = 1
        env_name = 'SawyerPushAndReachTorqueArenaTrainEnvBig-v0'
    else:
        raise NotImplementedError
    from robot.controller.her.test_sawyer_push import Wrapper

    return Wrapper(gym.make(env_name))


def test():
    from robot.utils.rl_utils import set_state
    env = make('pnr_torque')
    env.reset()

    state = env.state_vector()
    #print(env.observation_space, env.action_space)
    img = env.render(mode='rgb_array')
    cv2.imwrite('/home/hza/leap/push.png', img[:,:,::-1])

    env = make('pm')
    env.reset()
    img = env.render(mode='rgb_array')
    cv2.imwrite('/home/hza/leap/pm.png', img)
    exit(0)

    for i in tqdm.trange(100000):
        a = env.action_space.sample()
        img = env.render(mode='rgb_array')
        cv2.imshow('img', img)
        cv2.waitKey(1)
        o, r, done, info = env.step(a)
        print(r)

        if done:
            set_state(env, state)


if __name__ == '__main__':
    test()
