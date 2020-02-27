import torch
import numpy as np
import argparse
from robot.envs.sapien.exp.utils import set_state, get_state, eval_policy

from robot.controller.double_cem.pointmass_env import make
from robot.controller.double_cem.action_optimizer import ActionOptimizer
from robot.controller.double_cem.model import DynamicModel, NumpyWeightNetwork
from robot.controller.double_cem.double_cem import DoubleCEM


class Policy:
    def __init__(self, optimizer, N, horizon, env):
        self.optimizer = optimizer
        self.N = N
        self.horizon = horizon
        self.inp_dim = env.observation_space['observation'].shape[0]
        self.network = NumpyWeightNetwork(self.inp_dim, env.action_space.shape[0], 2, 32)
        self.state2obs = env.state2obs

        self.lb = env.action_space.low
        self.ub = env.action_space.high

        self.ob_lb = env.observation_space['observation'].low
        self.ob_ub = env.observation_space['observation'].high
        self.ob_std = (self.ob_ub - self.ob_lb)/4

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

        init_action = torch.tensor(np.stack([self.network.init_weights() for i in range(self.N * self.horizon)]),
                            dtype=torch.float).reshape(self.N, self.horizon, -1)
        action_std = init_action * 0 + 0.1**0.5

        s = [self.state2obs(state)] + [(self.ob_lb + self.ob_ub)/2 for i in range(self.N-1)]
        s_std = [self.ob_std * 0] + [self.ob_std for i in range(self.N-1)]

        s = torch.tensor(s, dtype=torch.float)
        s_std = torch.tensor(np.array(s_std), dtype=torch.float)
        self.optimizer.state = state.copy()
        _, _, actions, _ = self.optimizer(s, init_action, s_std, action_std)

        self.actions = actions.detach().cpu().numpy().reshape(self.N*self.horizon, -1)
        return self.__call__(state)


def constraint(s, t):
    d = ((s[:, None] - t[None, :])**2).sum(dim=-1)
    return torch.exp( d - 1 ).clamp(0, 40)
#for i in range(100):
#    print(i/10., constraint(torch.tensor([[0.]]), torch.tensor([[i/10]])))
#exit(0)


def main():
    #pass
    env_name = 'pm'
    model = DynamicModel(make, env_name, n=30)

    env = make(env_name)
    action_optimizer = ActionOptimizer(model, constraint, std=None,
                                #iter_num=20,
                                iter_num = 10,
                                num_mutation=400, num_elite=40,
                                alpha=0.1, trunc_norm=True,
                                )

    ob_space = env.observation_space['observation']
    optimizer = DoubleCEM(constraint, model, action_optimizer, 1,
                          num_mutation=100, num_elite=10, alpha=0.1,
                          upper_bound=ob_space.high, lower_bound=ob_space.low)

    #policy = Policy(optimizer, 4, 25, env)
    policy = Policy(optimizer, 1, 100, env)

    state = None
    eval_policy(policy, env, 12345, 10, 10, 'video{}.avi',
                use_hidden_state=True, progress_episode=True, timestep=100, start_state = state)


if __name__ == '__main__':
    main()
