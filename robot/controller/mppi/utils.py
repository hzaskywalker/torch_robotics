"""
Utility functions useful for TrajOpt algos
"""
import numpy as np
import multiprocessing as mp

def get_environment(env_name):
    from .humanoid_standup_env import HumanoidStandupEnv
    return HumanoidStandupEnv()

