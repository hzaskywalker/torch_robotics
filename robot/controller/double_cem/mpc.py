from robot.controller.cem import CEM
import torch


class MPC:
    # Real trajectory optimization

    def __init__(self, model, optimizer, state_space, action_space, state_std, action_std,
                 horizon, replan_period=1, add_actions=True, initial_iter=None, device='cuda:0'):
        assert 'rollout' in model.__dir__()
        self.model = model
        self.action_space = action_space
        self.state_space = state_space
        self.optimizer = optimizer

        self.horizon = horizon
        self.replan_period = replan_period
        self._iter = self.optimizer.iter_num

        self.state_std = state_std
        self.action_std = action_std

        if initial_iter is not None and initial_iter > 0:
            self.optimizer.iter_num = initial_iter

        if state_std is None:
            self.init_state_std = torch.tensor((state_space.high - state_space.low)/4, device=device, dtype=torch.float)
        else:
            self.init_state_std = torch.tensor(state_space.low * 0 + state_std, device=device, dtype=torch.float)

        if action_std is None:
            self.init_action_std = torch.tensor((action_space.high - action_space.low)/4, device=device, dtype=torch.float)
        else:
            self.init_action_std = torch.tensor(action_space.low * 0 + action_std, device=device, dtype=torch.float)

        self.device = device

    def init_actions(self, horizon):
        return

    def init_trajectory(self, N, horizon):
        states = torch.tensor([(self.state_space.high + self.state_space.low) * 0.5 for _ in range(horizon)],
                     dtype=torch.float, device=self.init_state_std.device)
        torch.tensor([(self.action_space.high + self.action_space.low) * 0.5 for _ in range(horizon)],
                     dtype=torch.float, device=self.init_action_std.device)
        return states,

    def set_model(self, model):
        self.model = model

    def reset(self):
        # random sample may be not good
        self.prev_trajectories = self.init_actions(self.horizon)
        self.ac_buf = None

    def __call__(self, obs):
        if self.ac_buf is not None:
            if self.ac_buf.shape[0] > 0:
                act, self.ac_buf = self.ac_buf[0], self.ac_buf[1:]
                return act
        obs = torch.tensor(obs, device=self.init_std.device, dtype=torch.float)
        self.prev_actions = self.optimizer(obs, self.prev_actions, self.init_std)
        self.optimizer.iter_num = self._iter

        self.ac_buf, self.prev_actions = torch.split(self.prev_actions, [self.replan_period, self.prev_actions.shape[0]-self.replan_period])
        if self.add_actions:
            self.prev_actions = torch.cat((self.prev_actions, self.init_actions(self.replan_period)))

        #import numpy as np
        #for i in range(1, len(self.prev_actions)):
        #    if np.random.random() < 0.2:
        #        self.prev_actions[i] = self.init_actions(1)[0]
        #k = self.prev_actions.shape[0]//2
        #self.prev_actions[k:] = self.init_actions(self.prev_actions.shape[0] - k)
        return self.__call__(obs).detach().cpu().numpy()
