import torch
import numpy as np
import argparse
from robot.envs.sapien.exp.utils import set_state, get_state, eval_policy

from robot.controller.double_cem.pointmass_env import make
from robot.controller.double_cem.action_optimizer import ActionOptimizer
from robot.controller.double_cem.model import DynamicModel, NumpyWeightNetwork


def add_parser(parser):
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--env_name', type=str, default='pm')
    parser.add_argument('--iter_num', type=int, default=20)
    parser.add_argument('--initial_iter', type=int, default=0)
    parser.add_argument('--num_mutation', type=int, default=400)
    parser.add_argument('--num_elite', type=int, default=40)
    parser.add_argument('--std', type=float, default=0.1 ** 0.5)
    parser.add_argument('--horizon', type=int, default=20)
    parser.add_argument('--num_proc', type=int, default=20)
    parser.add_argument('--video_num', type=int, default=1)
    parser.add_argument('--video_path', type=str, default='video{}.avi')
    parser.add_argument('--num_test', type=int, default=1)
    parser.add_argument('--timestep', type=int, default=100)
    parser.add_argument('--add_actions', type=int, default=1)
    parser.add_argument('--controller', type=str, default='cem', choices=['cem', 'poplin'])


class Policy:
    def __init__(self, optimizer, horizon, env):
        self.optimizer = optimizer
        self.horizon = horizon
        self.inp_dim = env.observation_space['observation'].shape[0]
        self.network = NumpyWeightNetwork(self.inp_dim, env.action_space.shape[0], 2, 32)
        self.state2obs = env.state2obs

        self.ub = env.action_space.high
        self.lb = env.action_space.low

    def reset(self):
        self.actions = None
        self.timestep = 0

    def network_control(self, obs, weights):
        #TODO: hack
        action = self.network(obs, weights)
        action = np.maximum(np.minimum(action, self.ub), self.lb)
        return action

    def __call__(self, state):
        if self.actions is not None:
            action, self.actions = self.actions[0], self.actions[1:]
            state = self.state2obs(state)
            return self.network_control(state, action)

        init = torch.tensor(np.stack([self.network.init_weights() for i in range(self.horizon)]), dtype=torch.float)
        state = torch.tensor(state, dtype=torch.float)
        self.actions = self.optimizer(state, init).detach().cpu().numpy()
        return self.__call__(state.detach().cpu().numpy())


def main():
    #pass
    parser = argparse.ArgumentParser()
    add_parser(parser)
    args = parser.parse_args()

    model = DynamicModel(make, args.env_name, n=args.num_proc)

    env = make(args.env_name)
    optimizer = ActionOptimizer(model, None, std=args.std,
                            iter_num=args.iter_num,
                            num_mutation=args.num_mutation, num_elite=args.num_elite,
                            alpha=0.1, trunc_norm=True,
                            #lower_bound=env.action_space.low, upper_bound=env.action_space.high
                            )

    policy = Policy(optimizer, args.timestep, env)

    state = None
    eval_policy(policy, env, 12345, args.num_test, args.video_num, args.video_path,
                use_hidden_state=True, progress_episode=True, timestep=args.timestep, start_state = state)


if __name__ == '__main__':
    main()
