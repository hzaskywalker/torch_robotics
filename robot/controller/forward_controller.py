import torch
from torch import nn

class ForwardControllerBase:
    def __init__(self, action_space, forward, cost, device='cuda:0'):
        self.action_space = action_space
        self.forward = forward
        self.cost = cost
        self.prev_actions = None
        self._device = device

    def __call__(self, x, cost, T):
        raise NotImplementedError


class GradidentDescentController(ForwardControllerBase):
    def __init__(self, T, action_space, forward, cost, optim_iter, optim_iter_init=None, lr=3., beta1=0.9, batch_size=1, device='cuda:0'):
        # need to predefine time length
        super(GradidentDescentController, self).__init__(action_space, forward, cost, device)
        self.T = T
        self.batch_size = batch_size
        self.optim_iter = optim_iter
        self.optim_iter_init = optim_iter if optim_iter_init is None else optim_iter
        self.lr = lr
        self.beta1 = beta1

    def reset(self):
        self.it = 0
        self.actions = nn.Parameter(torch.zeros((self.T, self.batch_size, self.action_space.shape[0]), device=self._device).to(self._device))
        self.optim = torch.optim.Adam([self.actions], lr=self.lr, betas=(self.beta1, 0.999))

    def rollout(self, x, actions):
        cost = 0
        for t in range(actions.shape[0]):
            t = self.forward(x, actions[t])
            cost = cost + self.cost(x, actions[t], t)
            x = t
        return cost, x

    def __call__(self, x, cost=None, T=20):
        """
        suppose x is batched
        """
        #self.forward.reset() # do we need to reset the forward model?
        iters = self.optim_iter if self.it > 0 else self.optim_iter_init
        for it in range(iters):
            self.optim.zero_grad()
            cost = self.rollout(x, self.actions)
            cost.backward()
            self.optim.step()
        self.it += 1
        return self.actions[0]
