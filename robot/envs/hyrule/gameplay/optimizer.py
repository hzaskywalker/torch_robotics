import torch
import numpy as np
from .simulator import Simulator
from robot.controller.cem import CEM

class Optimizer:
    def optimize(self, sim):
        raise NotImplementedError


from robot.utils.data_parallel import DataParallel


class Rollout:
    def __init__(self, make, horizon):
        self.sim: Simulator = make()
        self.horizon = horizon

    def __call__(self, s, a):
        rewards = []
        for s, a in zip(s, a):
            self.sim.load_state_vector(s)
            cost = 0
            for k in range(self.horizon):
                if len(a.shape) == 2:
                    self.sim.set_param(a[k])
                else:
                    self.sim.set_param(a)
                cost += self.sim.cost()
            rewards.append(cost)
        return np.array(rewards)

class SapienMujocoRolloutModel:
    def __init__(self, make, n=20, horizon=1):
        self.model = DataParallel(n, Rollout, make, horizon)

    def rollout(self, s, a):
        is_cuda =isinstance(a, torch.Tensor)
        device = 'cpu'
        if is_cuda:
            device = s.device
            s = s.detach().cpu().numpy()
            a = a.detach().cpu().numpy()
        r = self.model(s, a)
        if is_cuda:
            r = torch.tensor(r, dtype=torch.float, device=device)
        return r


class CEMOptimizer:
    def __init__(self, make_env, horizon, *args, num_proc=20, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.model = SapienMujocoRolloutModel(make_env, n=num_proc, horizon=horizon)
        self.cem = CEM(self.eval, *args, **kwargs)

    def eval(self, scene, a):
        s = scene[None, :].expand(a.shape[0], -1)
        return self.model.rollout(s, a)

    def optimize(self, sim: Simulator):
        state_vector = torch.tensor(sim.state_vector(), dtype=torch.float32)
        initial_action = torch.tensor(np.concatenate([i.data.reshape(-1) for i in sim.parameters]), dtype=torch.float32)
        output = self.cem(state_vector, initial_action).detach().cpu().numpy()
        sim.set_param(output)
