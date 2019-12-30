import os
import numpy as np
from robot.utils import togpu
import tqdm
os.environ['MUJOCO_GL'] = "osmesa" # for headless rendering
import gym
import cv2
from robot.controller import mb_controller
from robot.model.gnn_forward import GNNForwardAgent
from robot.utils.framebuffer import TrajBuffer
from robot.envs.dm_control import make as dm_make


def make(env_name, mode='Graph'):
    """
    :return: environment with graph, node attr and edge attr
    """
    task = {
        "Pendulum": "swingup",
        "Cheetah": "run",
        "Humanoid": "run"
    }
    return dm_make(env_name.lower(), task[env_name], mode=mode)


def visualize():
    """
    :return: visualize a trajectory given the environment.
    """
    pass


def train():
    """
    :return: training the environment with the trajectory
    """
    pass


def test_env_show():
    env:gym.Env = make('Pendulum', mode='')
    print(env.action_space, env.observation_space)
    env.reset()
    while True:
        img = env.render(mode='rgb_array') # or 'human
        a = env.action_space.sample()
        _, _, done, _ = env.step(a)
        if done:
            break


def test_graph_env():
    #env: gym.Env = make('Pendulum', mode='Graph')
    #env: gym.Env = make('Pendulum', mode='Graph')
    #import dm_control
    #env: gym.Env = make('Cheetah', mode='Graph')
    env: gym.Env = make('Cheetah', mode='Graph')
    x = env.reset()
    while True:
        a = env.action_space.sample()

        #img = env.render(mode='rgb_array') # or 'human
        #cv2.imwrite('x.jpg', img)
        #cv2.imshow('x.jgp', img)
        #cv2.waitKey(1)
        t, _, done, _ = env.step(a)
        if done:
            break


def test_gnn_model():
    env: gym.Env = make('Cheetah', mode='Graph')

    agent = GNNForwardAgent(0.01, env).cuda()

    s, a, t = [], [], []
    x = env.reset()
    for i in tqdm.trange(128):
        _a = env.action_space.sample()
        _t, _, _, _ = env.step(_a)
        s.append(x)
        a.append(_a)
        t.append(_t)
        x = _t
    s = togpu(np.array(s))
    a = togpu(np.array(a))
    t = togpu(np.array(t))
    for i in tqdm.trange(10000):
        agent.update(s, a, t)


def main():
    pass


if __name__ == '__main__':
    test_gnn_model()
