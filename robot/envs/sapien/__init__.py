from .control.cartpole import CartpoleEnv
from .control.halfcheetah import HalfCheetahEnv
from .control.humanoid import HumanoidEnv
from .control.pusher import PusherEnv
from .control.swimmer import SwimmerEnv
from .robotics.movo.movo_reach import MoveReachEnv
from .robotics.movo_xyz.movo_reach_xyz_env import MoveReachXYZEnv
from gym.wrappers import TimeLimit

def make(env_name):
    if env_name == 'cartpole':
        return TimeLimit(CartpoleEnv(), 200)
    elif env_name == 'halfcheetah':
        return TimeLimit(HalfCheetahEnv(), 1000)
    elif env_name == 'pusher':
        return TimeLimit(PusherEnv(), 100)
    elif env_name == 'humanoid':
        return HumanoidEnv()
    elif env_name == 'swimmer':
        return TimeLimit(SwimmerEnv(), 1000)
    elif env_name == 'movo_reach':
        return TimeLimit(MoveReachEnv('sparse'), 50)
    elif env_name == 'movo_reach_dense':
        return TimeLimit(MoveReachEnv('dense'), 50)
    elif env_name == 'movo_reach_xyz':
        return TimeLimit(MoveReachXYZEnv('sparse'), 50)
    elif env_name == 'table_world':
        from robot.envs.hyrule import Env
        return TimeLimit(Env(), 300)
    else:
        raise NotImplementedError

