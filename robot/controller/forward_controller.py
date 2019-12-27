import torch
import numpy as np
from torch import nn
from typing import Tuple

class ForwardControllerBase:
    def __init__(self, action_space, forward, cost, device='cuda:0'):
        self.action_space = action_space
        self.forward = forward
        self.cost = cost
        self.prev_actions = None
        self._device = device

    def set_model(self, model):
        self.forward = model

    def set_cost(self, cost):
        self.cost = cost

    def __call__(self, x, cost, T):
        raise NotImplementedError


class GDController(ForwardControllerBase):
    def __init__(self, T, action_space, forward, cost, optim_iter=10, optim_iter_init=100, lr=3., beta1=0.9, batch_size=1, device='cuda:0'):
        # need to predefine time length
        super(GDController, self).__init__(action_space, forward, cost, device)
        self.T = T
        self.batch_size = batch_size
        self.optim_iter = optim_iter
        self.optim_iter_init = optim_iter if optim_iter_init is None else optim_iter_init
        self.lr = lr
        self.beta1 = beta1
        self.actions = None

    def reset(self):
        self.it = 0
        self.actions = nn.Parameter(torch.zeros((self.T, self.batch_size, self.action_space.shape[0]), device=self._device).to(self._device))
        self.optim = torch.optim.Adam([self.actions], lr=self.lr, betas=(self.beta1, 0.999))

    def rollout(self, x, actions, start_it=0) -> Tuple[torch.Tensor, torch.Tensor]:
        cost: torch.Tensor = 0
        for it in range(start_it, actions.shape[0]):
            t = self.forward(x, actions[it])
            cost = self.cost(x, actions[it], t, it) + cost
            x = t
        return cost, x

    def __call__(self, x):
        """
        suppose x is batched
        """
        #self.forward.reset() # do we need to reset the forward model?
        if self.actions is None:
            self.reset()

        iters = self.optim_iter if self.it > 0 else self.optim_iter_init
        for it in range(iters):
            self.optim.zero_grad()
            cost, _ = self.rollout(x, self.actions, self.it)
            print(cost, _)
            cost.backward()
            self.optim.step()
        self.it += 1
        return self.actions[0]

    @staticmethod
    def test():
        from robot.envs.env_dx.cartpole import CartpoleDx
        from robot.envs import make
        from robot.utils import tocpu
        model = CartpoleDx()
        env = make('CartPole-v0')
        env.seed(0)

        obs = env.reset()
        T = 100

        def cost(s, a, t, it):
            x, dx, th, dth = torch.unbind(t, dim=1)
            if it == T-1 and False:
                cos_th, sin_th = torch.cos(th), torch.sin(th)
                s = (x**2) * 0.1 + (dx**2) * 0.1 + ((cos_th - 1) **2) + (sin_th**2) + (dth**2) * 0.1
                return s.sum()
            else:
                th_target = 20/360 *np.pi
                x_target = 2.2
                return (torch.nn.functional.relu(th - th_target) ** 2).sum() + \
                       (torch.nn.functional.relu(-th_target - (-th)) ** 2).sum() + \
                       (torch.nn.functional.relu(x - x_target) ** 2).sum() + \
                       (torch.nn.functional.relu(-x_target - (-x)) ** 2).sum()

        lr = 0.001
        controller = GDController(T, env.action_space, model,
                                  cost, optim_iter=10, optim_iter_init=100, lr=lr, batch_size=1)

        obs = torch.Tensor(obs)[None, :].cuda()
        x = tocpu(obs[0].view(-1))

        controller(obs)
        print(controller.rollout(obs, controller.actions))
        rewards = 0
        for i in controller.actions:
            u = float(i[0])
            x, r, d, _ = env.forward(x, u)
            rewards += r
            if d:
                break
        print(x, rewards)



if __name__:
    GDController.test()
