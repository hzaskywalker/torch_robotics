# code for cartpole
# TODO: wrap cartpole into a control environment.
# TODO: test various control algorithm.
# TODO: build cartpole in different environment.
import argparse
import numpy as np
import tqdm
import gym
from robot.envs.gym import make
from robot.utils import evaluate
from robot.controller.sac import sac
from robot.controller.td3 import td3
from robot.utils import tocpu


def main():
    #env = make('CartPole-v0', mode='dm_control')
    #env = make('CartPole-v0')
    env = make('MBRLCartpole-v0')
    state = env.reset()
    #controller = CEMController(5, 50, 5, env=env, std=2.)
    env._max_episode_steps = 200
    controller = sac(env, 20000, start_steps=10000)
    #controller = td3(env, 0, start_timesteps=1000, max_timesteps=20000)
    print(evaluate(env, controller, 200, 10))

if __name__ == '__main__':
    main()
