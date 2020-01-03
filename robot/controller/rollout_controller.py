# unlike forward controller
# we supppose that we have a very efficient rollout functions, which will return the observations and costs
from robot.controller.cem import CEM
import torch
from .cem import CEM

class RolloutCEMWrapper:
    def __init__(self, rollout, action_space, horizon, replan_period=1, device='cpu',
                 *args, **kwargs):
        self.rollout = rollout
        self.action_space = action_space
        self.rollout = rollout
        self.optimizer = CEM(self.cost, *args, **kwargs)

        self.horizon = horizon
        self.replan_period = replan_period
        self.init_var = torch.tensor((action_space.high - action_space.low)/16, device=device, dtype=torch.float)

    def cost(self, x, a):
        return self.rollout(x, a)[1] # ignore the cost and only use the reward

    def init_actions(self, horizon):
        return torch.tensor([self.action_space.sample() for _ in range(horizon)],
                            dtype=torch.float, device=self.init_var.device)

    def reset(self):
        # random sample may be not good
        self.prev_actions = self.init_actions(self.horizon)
        self.ac_buf = None

    def __call__(self, obs):
        if self.ac_buf is not None and self.ac_buf.shape[0] > 0:
            act, self.actions = self.actions[0], self.actions[1:]
            return act
        obs = torch.tensor(obs, device=self.init_var.device, dtype=torch.float)
        act = self.optimizer(obs, self.prev_actions, self.init_var)

        self.ac_buf, self.prev_actions = torch.split(self.prev_actions, self.replan_period)
        self.prev_actions = torch.cat((self.prev_actions, self.init_actions(self.horizon-self.replan_period)))
        return self.__call__(obs)



