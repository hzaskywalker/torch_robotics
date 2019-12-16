import gym
import numpy as np
from gym.spaces import Box
from gym import Wrapper

class CartPoleWrapper(Wrapper):
    def __init__(self, env):
        Wrapper.__init__(self, env)
        self.env.reset()
        self.env.action_space = Box(-10, 10, (1,))

    def forward(self, s, a):
        """
        :param s: state
        :param a: action
        :return: t, reward, done
        """
        # calculate the next state given s and a
        self.env.state = s
        self.unwrapped.force_mag = abs(np.clip(float(a), -10, 10))
        return self.env.step(1 if a > 0 else 0)[:3]
