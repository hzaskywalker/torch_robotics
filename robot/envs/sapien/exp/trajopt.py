import gym
import os
import torch
from robot.controller.td3 import td3
from robot.envs.sapien.exp.utils import make

import argparse
import numpy as np
from robot.controller.rollout_controller import RolloutCEM
from robot.envs.sapien.exp.utils import set_state, eval_policy
from robot.utils.data_parallel import DataParallel

class Rollout:
    def __init__(self, make, env_name):
        self.env = make(env_name).unwrapped #discard the other things...

    def __call__(self, s, a):
        rewards = []
        for s, a in zip(s, a):
            set_state(self.env, s)
            reward = []
            for action in a:
                _, r, _, _ = self.env.step(action)
                reward.append(-r)
            rewards.append(reward)
        return np.array(rewards)

class SapienMujocoRolloutModel:
    def __init__(self, env_name, n=20):
        self.model = DataParallel(n, Rollout, make, env_name)

    def rollout(self, s, a):
        is_cuda =isinstance(a, torch.Tensor)
        if is_cuda:
            device = s.device
            s = s.detach().cpu().numpy()
            a = a.detach().cpu().numpy()
        r = self.model(s, a)
        if is_cuda:
            r = torch.tensor(r, dtype=torch.float, device=device)
        return None, r.sum(dim=1)

def main():
    #pass
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--env_name', type=str, default='halfcheetah')
    parser.add_argument('--iter_num', type=int, default=5)
    parser.add_argument('--num_mutation', type=int, default=100)
    parser.add_argument('--num_elite', type=int, default=10)
    parser.add_argument('--std', type=float, default=0.2)
    parser.add_argument('--horizon', type=int, default=20)
    parser.add_argument('--num_proc', type=int, default=20)
    parser.add_argument('--video_num', type=int, default=1)
    parser.add_argument('--video_path', type=str, default='video{}.avi')
    parser.add_argument('--num_test', type=int, default=10)
    args = parser.parse_args()

    model = SapienMujocoRolloutModel(args.env_name, n=args.num_proc)

    env = make(args.env_name)
    controller = RolloutCEM(model, action_space=env.action_space, horizon=args.horizon, iter_num=args.iter_num, num_mutation=args.num_mutation, num_elite=args.num_elite,
                            alpha=0.1, trunc_norm=2, lower_bound=env.action_space.low, upper_bound=env.action_space.high)

    eval_policy(controller, env, 12345, args.num_test, args.video_num, args.video_path, use_hidden_state=True, progress_episode=True)


if __name__ == '__main__':
    main()
