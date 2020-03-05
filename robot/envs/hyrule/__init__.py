from .simulator import Simulator
from .rl_env import ArmReach
from .loader import load_scene, load_json, dump_json

def make(filepath, timestep=50):
    from gym.wrappers import TimeLimit
    return TimeLimit(ArmReach('dense'), timestep)
    env = Simulator()
    params = load_json(filepath)
    load_scene(env, params)
    return TimeLimit(env, timestep)
