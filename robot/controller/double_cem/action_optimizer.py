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
        if len(x.shape) == 1:
            x = x[None, :].expand(a.shape[0], -1)
            xx = None
        else:
            #xx = int(x.shape[0])
            xx = 40
            x = x[torch.randint(x.shape[0], size=(a.shape[0] * xx,))]
            a = a[:, None].expand(-1, xx, -1, -1)
            a = a.reshape(-1, *a.shape[2:])

        timestep = [a.shape[1] for i in range(a.shape[0])]
        t, r = self.model(x, a, timestep=timestep)
        if not isinstance(r, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float, device=x.device)
            r = torch.tensor(r, dtype=torch.float, device=x.device)

        if self.targets is not None:
            # TODO: only consider the last reward ...
            r = (self.constraint(t, self.targets) + self.value[None, :]).min(dim=1)[0] # choose the minimum among values

        if xx is not None:
            r = r.reshape(-1, xx).mean(dim=1)
        #print(r.mean())
        return r

    def __call__(self, scene, mean=None, std=None, targets=None, value=None, show_progress=False, return_std=False):
        self.targets = targets
        self.value = value
        #print(self.value)
        return super(ActionOptimizer, self).__call__(scene, mean, std, show_progress, return_std=return_std)
