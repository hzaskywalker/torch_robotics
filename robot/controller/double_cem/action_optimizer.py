# we use the action optimizer based on a non-linear
import torch
from robot.controller.cem import CEM


class ActionOptimizer(CEM):
    def __init__(self, model, constraint, *args, **kwargs):
        self.targets = None
        self.value = None
        self.constraint = constraint
        self.model = model
        super(ActionOptimizer, self).__init__(self.reward, *args, **kwargs)

    def reward(self, x, a):
        x = x[None, :].expand(a.shape[0], -1)
        timestep = [a.shape[1] for i in range(a.shape[0])]
        t, r = self.model(x, a, timestep=timestep)
        if not isinstance(r, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float, device=x.device)
            r = torch.tensor(r, dtype=torch.float, device=x.device)

        if self.targets is not None:
            raise NotImplementedError
            r = (self.constraint(t, self.targets) + self.value[None,:]).min(dim=1) # choose the minimum among values
        return r

    def __call__(self, scene, mean=None, std=None, targets=None, value=None, show_progress=False):
        self.targets = targets
        self.value = value
        return super(ActionOptimizer, self).__call__(scene, mean, std, show_progress)
