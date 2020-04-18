from gym.wrappers import TimeLimit


def make(env_name):
    if env_name == 'acrobat2':
        from .acrobat import GoalAcrobat
        return TimeLimit(GoalAcrobat(), 50)
    elif env_name == 'arm':
        from ....envs.hyrule.rl_env import ArmReachWithXYZ
        return TimeLimit(ArmReachWithXYZ(), 100)
    elif env_name == 'plane':
        from robot.envs.toy.plane import GoalPlane
        return TimeLimit(GoalPlane(), 50)
    elif env_name == 'diff_acrobat':
        from robot.envs.diff_phys.acrobat import GoalAcrobat
        return TimeLimit(GoalAcrobat(), 50)
    else:
        raise NotImplementedError
