# unlike forward controller
# we supppose that we have a very efficient rollout functions, which will return the observations and costs
from robot.controller.cem import CEM
import torch
from .cem import CEM

class RolloutCEM:
    def __init__(self, model, action_space, horizon, replan_period=1, add_actions=True, initial_iter=None,
                 device='cuda:0', std=None, *args, **kwargs):
        # add actions=0 means we plan for the whole trajectory
        assert 'rollout' in model.__dir__()
        self.model = model
        self.action_space = action_space
        self.optimizer = CEM(self.cost, *args, **kwargs)

        self.horizon = horizon
        self.replan_period = replan_period
        self.add_actions = add_actions
        self._iter = self.optimizer.iter_num
        if initial_iter is not None and initial_iter > 0:
            self.optimizer.iter_num = initial_iter

        if std is None:
            self.init_std = torch.tensor((action_space.high - action_space.low)/4, device=device, dtype=torch.float)
        else:
            self.init_std = torch.tensor(action_space.low * 0 + std, device=device, dtype=torch.float)
        self.device = device

    def cost(self, x, a):
        x = x[None, :].expand(a.shape[0], -1)
        if self.goal is None:
            # not goal conditioned
            out = self.model.rollout(x, a)[1]
        else:
            out = self.model.rollout(x, a, self.goal)[1]

        if not isinstance(out, torch.Tensor):
            out = torch.tensor(out, dtype=torch.float, device=self.device)
        return out


    def init_actions(self, horizon):
        return torch.tensor([(self.action_space.high + self.action_space.low) * 0.5 for _ in range(horizon)],
                            dtype=torch.float, device=self.init_std.device)

    def set_model(self, model):
        self.model = model

    def reset(self):
        # random sample may be not good
        self.prev_actions = self.init_actions(self.horizon)
        self.ac_buf = None

    def __call__(self, obs):
        if self.ac_buf is not None:
            if self.ac_buf.shape[0] > 0:
                act, self.ac_buf = self.ac_buf[0], self.ac_buf[1:]
                return act

        if isinstance(obs, dict):
            obs, goal = obs['observation'], obs['desired_goal']
            obs = torch.tensor(obs, device=self.init_std.device, dtype=torch.float)
            self.goal = torch.tensor(goal, device=self.init_std.device, dtype=torch.float)
        else:
            obs = torch.tensor(obs, device=self.init_std.device, dtype=torch.float)
            self.goal = None
        self.prev_actions = self.optimizer(obs, self.prev_actions, self.init_std)
        self.optimizer.iter_num = self._iter

        self.ac_buf, self.prev_actions = torch.split(self.prev_actions, [self.replan_period, self.prev_actions.shape[0]-self.replan_period])
        if self.add_actions:
            self.prev_actions = torch.cat((self.prev_actions, self.init_actions(self.replan_period)))

        return self.__call__(obs).detach().cpu().numpy()
