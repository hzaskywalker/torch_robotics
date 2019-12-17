# code for cartpole
# TODO: wrap cartpole into a control environment.
# TODO: test various control algorithm.
# TODO: build cartpole in different environment.
import argparse
import numpy as np
import tqdm
import gym
from robot.envs import make
from robot.controller.cem import CEMController
from robot.controller.sac import sac
from robot.controller.td3 import td3
from robot.utils import tocpu

def evaluate(env: gym.Env, controller, timestep=200, num_episode=10, horizon=20):
    ans = []
    for i in tqdm.trange(num_episode):
        state = env.reset()
        total = 0
        for j in tqdm.trange(timestep):
            #action = env.action_space.sample()
            action = controller(state)
            state, r, d, _ = env.step(action)
            total += r
            if d:
                break
        ans.append(total)
    return np.mean(ans)

def main():
    #env = make('CartPole-v0', mode='dm_control')
    env = make('CartPole-v0')
    state = env.reset()
    #controller = CEMController(5, 50, 5, env=env, std=2.)
    env._max_episode_steps = 200
    #controller = sac(env, 20000, start_steps=10000)
    controller = td3(env, 0, start_timesteps=1000, max_timesteps=20000)
    print(evaluate(env, controller, 200, 10, horizon=20))

if __name__ == '__main__':
    main()
