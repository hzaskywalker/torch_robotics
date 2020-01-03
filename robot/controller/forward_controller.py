import torch
import numpy as np
from torch import nn
from typing import Tuple
from robot.controller.cem import CEM


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

    def reset(self):
        raise NotImplementedError

    def control(self, x):
        raise NotImplementedError

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x).to(self._device)
        squeeze = (x.dim() == 1)
        if squeeze:
            x = x[None, :]
        action = self.control(x)
        if squeeze:
            return action[0]
        else:
            return action


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

    def control(self, x):
        """
        suppose x is batched
        """
        iters = self.optim_iter if self.it > 0 else self.optim_iter_init
        for it in range(iters):
            self.optim.zero_grad()
            cost, _ = self.rollout(x, self.actions, self.it)
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


class CEMController(ForwardControllerBase):
    def __init__(self, timestep, action_space, forward_model, cost, std,
                 iter_num, num_mutation, num_elite, mode='fix', device='cuda:0', *args, **kwargs):
        super(CEMController, self).__init__(action_space, forward_model, cost, device)
        self.timestep = timestep
        self.cem = CEM(eval_function=self.eval_function, iter_num=iter_num, num_mutation=num_mutation, num_elite=num_elite, std=std, *args, **kwargs)
        self.mode = mode

    def eval_function(self, x, action, start_it=0):
        assert x.shape[0] == 1
        x = x.expand(action.shape[0], *x.shape[1:])
        cost: torch.Tensor = 0
        for it in range(len(action[0])):
            t = self.forward(x, action[:, it])
            cost = self.cost(x, action[:, it], t, it + start_it) + cost
            x = t
        return cost

    def initial_action(self):
        if self.mode == 'fix':
            return torch.zeros((self.timestep, *self.action_space.shape), device=self._device)
        else:
            raise NotImplementedError

    def reset(self):
        pass

    def control(self, x):
        """
        suppose x is batched
        """
        mean = self.initial_action()
        out = self.cem(x, mean)
        return out[0:1]


    @staticmethod
    def test():
        from robot.envs.env_dx.cartpole import CartpoleDx
        from robot.envs import make
        from robot.utils import tocpu, evaluate
        model = CartpoleDx()
        env = make('CartPole-v0')
        env.seed(0)

        T = 200

        def cost(s, a, t, it):
            x, dx, th, dth = torch.unbind(t, dim=1)
            th_target = 20/360 *np.pi
            x_target = 2.2
            """
            out = (torch.nn.functional.relu(th - th_target) ** 2) + \
                   (torch.nn.functional.relu(-th_target - (-th)) ** 2) + \
                   (torch.nn.functional.relu(x - x_target) ** 2) + \
                   (torch.nn.functional.relu(-x_target - (-x)) ** 2)
                   """
            out = ((th - 0) ** 2) + ((th_target - 0) ** 2) # this loss is much easier to optimize
            return out

        controller = CEMController(20, env.action_space, model,
                                  cost, std=3.,
                                  iter_num=5, num_mutation=100, num_elite=10,
                                  mode='fix')

        print('evaluating...')
        print(evaluate(env, controller, timestep=T, use_tqdm=True))



if __name__ == '__main__':
    #GDController.test()
    CEMController.test()
