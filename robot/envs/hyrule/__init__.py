from .simulator import Simulator
from .loader import load_scene, load_waypoints, load_json, dump_json

def make(filepath, timestep=100):
    from gym.wrappers import TimeLimit
    env = Simulator()
    params = load_json(filepath)
    load_scene(env, params)
    return TimeLimit(env, timestep)
