import torch

import argparse
import numpy as np
from robot.controller.rollout_controller import RolloutCEM
from robot.envs.sapien.exp.utils import set_state, get_state, eval_policy
from robot.utils.data_parallel import DataParallel


from robot.controller.double_cem.pointmass_env import make

class Rollout:
    def __init__(self, make, env_name):
        self.env = make(env_name)

    def __call__(self, s, a):
        rewards = []
        for s, a in zip(s, a):
            set_state(self.env, s)
            reward = 0
            for action in a:
                s, r, done, _ = self.env.step(action)
                reward += -r
                if done:
                    break
            rewards.append(reward)
        return np.array(rewards)

class SapienMujocoRolloutModel:
    def __init__(self, make, env_name, n=20):
        self.model = DataParallel(n, Rollout, make, env_name)

    def rollout(self, s, a):
        is_cuda =isinstance(a, torch.Tensor)
        if is_cuda:
            s = s.detach().cpu().numpy()
            a = a.detach().cpu().numpy()
        r = self.model(s, a)
        if is_cuda:
            r = torch.tensor(r, dtype=torch.float, device='cuda:0')
        return None, r


def add_parser(parser):
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--env_name', type=str, default='pm')
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

    model = SapienMujocoRolloutModel(make, args.env_name, n=args.num_proc)

    env = make(args.env_name)
    print(env.action_space.low, env.action_space.high)
    controller = RolloutCEM(model, action_space=env.action_space,
                            add_actions=args.add_actions,
                            horizon=args.horizon, std=args.std,
                            iter_num=args.iter_num,
                            initial_iter=args.initial_iter,
                            num_mutation=args.num_mutation, num_elite=args.num_elite, alpha=0.1, trunc_norm=True, lower_bound=env.action_space.low, upper_bound=env.action_space.high)

    state = None
    eval_policy(controller, env, 12345, args.num_test, args.video_num, args.video_path, use_hidden_state=True, progress_episode=True, timestep=args.timestep, start_state = state)


if __name__ == '__main__':
    main()
