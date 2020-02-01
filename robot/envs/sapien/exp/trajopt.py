import gym
import os
import torch
from robot.controller.td3 import td3
from robot.envs.sapien.exp.utils import make

import argparse
import numpy as np
from robot.controller.rollout_controller import RolloutCEM
from robot.envs.sapien.exp.utils import set_state, get_state, eval_policy
from robot.utils.data_parallel import DataParallel

class Rollout:
    def __init__(self, make, env_name):
        self.env = make(env_name).unwrapped #discard the other things...

    def __call__(self, s, a):
        rewards = []
        obs = []
        for s, a in zip(s, a):
            set_state(self.env, s)
            reward = []
            for action in a:
                s, r, _, _ = self.env.step(action)
                reward.append(-r)
            rewards.append(reward)
            obs.append(s)
        return np.array(obs), np.array(rewards)

class SapienMujocoRolloutModel:
    def __init__(self, env_name, n=20):
        self.model = DataParallel(n, Rollout, make, env_name)

    def rollout(self, s, a):
        single_step = len(a.shape) > 2
        is_cuda =isinstance(a, torch.Tensor)
        if is_cuda:
            device = s.device
            s = s.detach().cpu().numpy()
            a = a.detach().cpu().numpy()
        obs, r = self.model(s, a)
        if is_cuda:
            obs = torch.tensor(obs, dtype=torch.float, device=device)
            r = torch.tensor(r, dtype=torch.float, device=device)
        return obs, r.sum(dim=1)


def add_parser(parser):
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--env_name', type=str, default='halfcheetah')
    parser.add_argument('--iter_num', type=int, default=5)
    parser.add_argument('--initial_iter', type=int, default=0)
    parser.add_argument('--num_mutation', type=int, default=100)
    parser.add_argument('--num_elite', type=int, default=10)
    parser.add_argument('--std', type=float, default=0.5)
    parser.add_argument('--horizon', type=int, default=20)
    parser.add_argument('--num_proc', type=int, default=20)
    parser.add_argument('--video_num', type=int, default=1)
    parser.add_argument('--video_path', type=str, default='video{}.avi')
    parser.add_argument('--num_test', type=int, default=1)
    parser.add_argument('--timestep', type=int, default=1000)
    parser.add_argument('--add_actions', type=int, default=1)
    parser.add_argument('--controller', type=str, default='cem', choices=['cem', 'poplin'])


def main():
    #pass
    parser = argparse.ArgumentParser()
    add_parser(parser)
    args = parser.parse_args()

    model = SapienMujocoRolloutModel(args.env_name, n=args.num_proc)

    env = make(args.env_name)
    print(env.action_space.low, env.action_space.high)
    controller = RolloutCEM(model, action_space=env.action_space,
                            add_actions=args.add_actions,
                            horizon=args.horizon, std=args.std,
                            iter_num=args.iter_num,
                            initial_iter=args.initial_iter,
                            num_mutation=args.num_mutation, num_elite=args.num_elite, alpha=0.1, trunc_norm=True, lower_bound=env.action_space.low, upper_bound=env.action_space.high)

    #state = np.array([1.62296279, -0.47857534, 2.38391795, -1.74633455, -1.11519682, 0.06964599, 0.0770168, 0.12503629, 0.0168354 - 0.27590335])
    #state = np.array([ 1.62756207,-0.42829709,2.23247287,-1.71191108,-1.55460429, 0.1903247, 0.08096971, 0.115363,-0.00278995, 0.52407408])
    #state = np.array([3.152642,0.167744,2.529410,-1.748099,-0.941260,0.156150,-0.095661,-0.249043,0.056756,0.874108])
    #state = np.array([ 4.413944,0.272971,1.681765,-1.713469,1.360272,-1.279943,-0.349565,-1.632567,1.230526,1.414734])
    #state = np.array([0.,-0.101701,2.330995,-1.751561,-0.030056,-0.396141,-0.773034,-1.187929,0.042890,3.202599])
    #state = np.array([4.327683,0.181096,2.222121,-1.746923,-0.479730,-0.029378,-0.211866,-0.374511,-0.020633,1.233133])
    #state = np.array([4.329239,-0.167083,2.249954,-1.718177,-0.538452,0.279116,-0.071928,0.033003,-0.777342,2.174717])
    state = None
    eval_policy(controller, env, 12345, args.num_test, args.video_num, args.video_path, use_hidden_state=True, progress_episode=True, timestep=args.timestep, start_state = state)


if __name__ == '__main__':
    main()
