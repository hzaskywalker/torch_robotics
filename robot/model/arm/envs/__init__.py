from .acrobat import GoalAcrobat
from gym.wrappers import TimeLimit
from ....envs.hyrule.rl_env import ArmReachWithXYZ
from robot.envs.toy.plane import GoalPlane
from robot.envs.diff_phys.acrobat import GoalAcrobat


def make(env_name):
    if env_name == 'acrobat2':
        return TimeLimit(GoalAcrobat(), 50)
    elif env_name == 'arm':
        return TimeLimit(ArmReachWithXYZ(), 50)
    elif env_name == 'plane':
        return TimeLimit(GoalPlane(), 50)
    elif env_name == 'diff_acrobat':
        return TimeLimit(GoalAcrobat(), 50)
    else:
        raise NotImplementedError
