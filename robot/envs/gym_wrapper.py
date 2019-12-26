import gym
import numpy as np
from gym.spaces import Box
from gym import Wrapper

class CartPoleWrapper(Wrapper):
    def __init__(self, env:gym.Env):
        Wrapper.__init__(self, env.unwrapped)
        self.action_space = Box(-10, 10, (1,))

    def seed(self, seed=None):
        self.env.seed(seed)
        self.action_space.seed(seed)

    def forward(self, s, a):
        """
        :param s: state
        :param a: action
        :return: t, reward, done
        """
        # calculate the next state given s and a
        self.unwrapped.reset()
        self.unwrapped.state = s
        return self.step(a)

    def step(self, a):
        self.unwrapped.force_mag = abs(np.clip(float(a), -10, 10))
        return self.env.step(1 if a > 0 else 0)
