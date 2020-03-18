from .acrobat import GoalAcrobat
from gym.wrappers import TimeLimit
from ....envs.hyrule.rl_env import ArmReachWithXYZ


def make(env_name):
    if env_name == 'acrobat2':
        return TimeLimit(GoalAcrobat(), 50)
    elif env_name == 'arm':
        return TimeLimit(ArmReachWithXYZ(), 50)
    else:
        raise NotImplementedError
