import gym
import os
import torch
from robot.controller.td3 import td3
from robot.envs.sapien.exp.utils import make

import argparse
import numpy as np
from robot.controller.poplin import PoplinController, WeightNetwork
from robot.envs.sapien.exp.utils import set_state, get_state, eval_policy
from robot.utils.data_parallel import DataParallel
from robot.utils import as_input

from robot.envs.sapien.exp.trajopt import add_parser

class NumpyWeightNetwork:
    def __init__(self, in_feature, out_feature, num_layers, mid_channels, device='cuda:0'):
        layers = []
        if num_layers == 1:
            layers.append((in_feature, out_feature, False))
        else:
            if isinstance(mid_channels, int):
                mid_channels = [mid_channels] * (num_layers-1)
            assert len(mid_channels) == num_layers - 1
            layers.append((in_feature, mid_channels[0], True))
            for i in range(num_layers-2):
                layers.append((mid_channels[i], mid_channels[i+1], True))
            layers.append((mid_channels[num_layers-2], out_feature, False))

        self.in_feature = in_feature
        self.tanh = np.tanh
        self.num_layer = num_layers
        self.layers = layers
        self._device = device

        start = 0
        self.w = []
        self.b = []
        self.scale = []

        for i, o, _ in self.layers:
            r = start + i * o
            self.w.append(slice(start, r))
            start = r

            r = start + o
            self.b.append(slice(start, r))
            start = r

    def __call__(self, x, weights):
        for l in range(self.num_layer):
            i, o, tanh = self.layers[l]
            w = weights[self.w[l]]
            b = weights[self.b[l]]

            w = w.reshape(i, o)
            b = b.reshape(1, o)
            x = x @ w + b
            if tanh:
                x = self.tanh(x)
        return x

class Rollout:
    def __init__(self, make, env_name, num_layer, mid_channel):
        self.env = make(env_name).unwrapped  # discard the other things...
        inp_dim = self.env.observation_space.shape[0]
        oup_dim = self.env.action_space.shape[0]
        self.network = NumpyWeightNetwork(inp_dim, oup_dim, num_layer, mid_channel)
        self.lb = self.env.action_space.low
        self.ub = self.env.action_space.high

    def __call__(self, s, a, timestep):
        rewards = []
        obs = []
        for s, a, timestep in zip(s, a, timestep):
            set_state(self.env, s)

            s = self.env._get_obs()

            reward = []
            #for action in a:
            for i in range(timestep):
                if len(a.shape)>1:
                    action = a[i]
                else:
                    action = a

                action = self.network(s, action)
                action = np.maximum(np.minimum(action, self.ub), self.lb)

                s, r, _, _ = self.env.step(action)
                reward.append(-r)
            rewards.append(reward)
            obs.append(s)
        return np.array(obs), np.array(rewards)

class SapienMujocoRolloutModel:
    def __init__(self, env_name, n=20, *args, **kwargs):
        self.model = DataParallel(n, Rollout, make, env_name, *args, **kwargs)

    def rollout(self, s, a, timestep):
        is_cuda = isinstance(a, torch.Tensor)
        if is_cuda:
            device = s.device
            s = s.detach().cpu().numpy()
            a = a.detach().cpu().numpy()
        obs, r = self.model(s, a, timestep)
        if is_cuda:
            obs = torch.tensor(obs, dtype=torch.float, device=device)
            r = torch.tensor(r, dtype=torch.float, device=device)
        return obs, r.sum(dim=1)


class MyPoplin(PoplinController):
    def __init__(self, *args, mode='sep', inp_dim=None, oup_dim=None, num_layers=2, mid_channels=32, **kwargs):
        super(MyPoplin, self).__init__(*args, inp_dim=inp_dim, oup_dim=oup_dim,
                                       num_layers=num_layers, mid_channels=mid_channels, **kwargs)
        self._cur_weights = self.network.init_weights()
        self.network = NumpyWeightNetwork(inp_dim, oup_dim, num_layers, mid_channels)
        self.lb = self.lb.detach().cpu().numpy()
        self.ub = self.ub.detach().cpu().numpy()
        self.mode = mode

    def reset(self):
        self.cur_weights = self._cur_weights
        if self.mode == 'sep':
            super(MyPoplin, self).reset()
        self.cur_weights = self._cur_weights

    def rollout(self, obs, weights):
        obs = obs.expand(weights.shape[0], -1) # (500, x)
        timestep = np.zeros((obs.shape[0],), dtype=np.int32) + self.current_horizon
        _, reward = self.model.rollout(obs, weights, timestep)
        return reward

    def network_control(self, obs, weights):
        #TODO: hack
        action = self.network(obs[0, -self.network.in_feature:].detach().cpu().numpy(), weights.detach().cpu().numpy())
        action = np.maximum(np.minimum(action, self.ub), self.lb)
        return torch.tensor(action)

    @as_input(2, classmethod=True)
    def __call__(self, obs):
        self.current_horizon = self.horizon
        if self.mode == 'sep':
            return super(MyPoplin, self).__call__(obs)
        else:
            self.cur_weights = self.cem(obs, self.cur_weights)
            return self.network_control(obs, self.cur_weights)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--env_name', type=str, default='halfcheetah')
    parser.add_argument('--iter_num', type=int, default=5)
    parser.add_argument('--num_mutation', type=int, default=100)
    parser.add_argument('--num_elite', type=int, default=10)
    parser.add_argument('--std', type=float, default=0.5)
    parser.add_argument('--horizon', type=int, default=20)
    parser.add_argument('--num_proc', type=int, default=20)
    parser.add_argument('--video_num', type=int, default=1)
    parser.add_argument('--video_path', type=str, default='video_poplin{}.avi')
    parser.add_argument('--num_test', type=int, default=1)
    parser.add_argument('--timestep', type=int, default=1000)
    parser.add_argument('--controller', type=str, default='cem', choices=['cem', 'poplin'])
    parser.add_argument('--mode', type=str, default='sep', choices=['sep', 'u'])
    args = parser.parse_args()

    model = SapienMujocoRolloutModel(args.env_name, n=args.num_proc, num_layer=2, mid_channel=32)

    env = make(args.env_name)
    print(env.action_space.low, env.action_space.high)
    controller = MyPoplin(model,
                          extension=None, # use the default cost
                          mode=args.mode,
                          horizon=args.horizon,
                          inp_dim=env.observation_space.shape[0],
                          oup_dim=env.action_space.shape[0],
                          iter_num=args.iter_num,
                          num_mutation=args.num_mutation,
                          num_elite=args.num_elite,
                          action_space=env.action_space,
                          trunc_norm=True,
                          std=0.1 ** 0.5,
                          num_layers=2,
                          )

    #env.reset()
    #s = get_state(env)
    #controller.reset()
    #controller(s)

    state = None
    eval_policy(controller, env, 12345, args.num_test, args.video_num, args.video_path, use_hidden_state=True, progress_episode=True, timestep=args.timestep, start_state = state)

if __name__ == '__main__':
    main()

