import gym
from .gym_wrapper import CartPoleWrapper
from .dm_control import make as dm_make

def make(env_name, mode='gym'):
    if env_name == 'CartPole-v0':
        if mode == 'gym':
            return CartPoleWrapper(gym.make(env_name))
        elif mode == 'dm_control':
            return dm_make('cartpole', task_name='balance')
    else:
        raise NotImplementedError
