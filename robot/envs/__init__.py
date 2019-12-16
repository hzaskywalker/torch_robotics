import gym
from .gym_wrapper import CartPoleWrapper

def make(env_name):
    if env_name == 'CartPole-v0':
        return CartPoleWrapper(gym.make(env_name))
    else:
        raise NotImplementedError
