from .cartpole import CartpoleEnv
from .halfcheetah import HalfCheetahEnv
from .humanoid import HumanoidEnv
from .pusher import PusherEnv
from .swimmer import SwimmerEnv
from gym.wrappers import TimeLimit

def make(env_name):
    if env_name == 'cartpole':
        return TimeLimit(CartpoleEnv(), 200)
    elif env_name == 'halfcheetah':
        return HalfCheetahEnv()
    elif env_name == 'humanoid':
        return HumanoidEnv()
    elif env_name == 'pusher':
        return PusherEnv()
    elif env_name == 'swimmer':
        return SwimmerEnv()
    else:
        raise NotImplementedError

