# controller for forward dynamics
import numpy as np
import tqdm
import torch
from robot.utils.trunc_norm import trunc_norm


class DoubleCEM:
    def __init__(self, constraint, rollout, optimizer, iter_num, num_mutation=100, num_elite=10,
                 alpha=0., upper_bound=None, lower_bound=None, trunc_norm=False, inf=int(1e9), env=None):

        self.constraint = constraint # the constraint function measures the loss if two states doesn't match..
        self.rollout = rollout

        self.optimizer = optimizer

        self.iter_num = iter_num
        self.num_mutation = num_mutation
        self.num_elite = num_elite
        self.alpha = alpha
        self.trunc_norm = trunc_norm
        self.inf = inf

        if upper_bound is not None:
            upper_bound = torch.tensor(upper_bound, dtype=torch.float)
        if lower_bound is not None:
            lower_bound = torch.tensor(lower_bound, dtype=torch.float)
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        self.state = None
        self.env = env


    def population2state(self, pop):
        #TODO: hack here
        return torch.cat((pop[..., :1]*0, pop, pop * 0 + torch.tensor(self.state[-2:], dtype=torch.float)), dim=-1)


    def value(self, populations, actions, targets=None, target_values=None, min_dim=1):
        """
        if min_dim == 1: return the reward of populations, otherwise, return the reward of targets ...
        """
        if len(actions.shape) == 2:
            timestep = np.zeros((populations.shape[0],), dtype=np.int32) + actions.shape[0]
            actions = actions[None, :].expand(populations.shape[0], -1, -1)
        else:
            raise NotImplementedError
        t, r = self.rollout(self.population2state(populations), actions, timestep)
        self._reached = t
        if targets is not None:
            # TODO: only consider the last reward
            r = (self.constraint(t, targets) + target_values[None, :]).min(dim=min_dim)[0]  # choose the minimum among values
        return r

    def print_value(self, targets, values, name):
        import cv2
        img = np.zeros((512, 512, 3), dtype=np.float32)
        for a, b in zip(targets, values):
            x, y = a
            x = int((x + 4)/8 * 512)
            y = int((y + 4)/8 * 512)

            color = int((10-b)/10 * 255)

            cv2.circle(img, (x, y), 3, (color, color, color), -1)
        cv2.imwrite(name, img)

    def __call__(self, states, actions, states_std, actions_std=None, show_progress=True):
        # scene is the first states...
        # states (N, state_dim)
        # the first is the initial states

        N = len(states)
        if isinstance(states_std, float):
            states_std = states * 0 + states_std

        shape = (states.shape[0], self.num_mutation, *states.shape[1:])
        values = torch.zeros((states.shape[0], self.num_mutation), device=states.device)

        with torch.no_grad():
            ran = tqdm.trange if show_progress else range
            for iter_ in ran(self.iter_num):
                # pass
                mean = states
                _std = states_std
                if self.lower_bound is not None and self.upper_bound is not None:
                    lb_dist = mean - self.lower_bound.to(mean.device)
                    ub_dist = -mean + self.upper_bound.to(mean.device)
                    _std = torch.min(torch.abs(torch.min(lb_dist, ub_dist) / 2), states_std)

                _std[0] = 0 # the first state will never change...

                if self.trunc_norm:
                    noise = trunc_norm(shape, device=mean.device)
                else:
                    noise = torch.randn(shape, device=mean.device)
                populations = noise * _std[:, None] + mean[:, None]
                assert noise.shape == populations.shape


                targets, value = None, None
                for i in tqdm.trange(N-1, -1, -1):
                    for j in range(len(populations[i])):
                        while self.env.unwrapped._position_inside_wall(populations[i][j]):
                            populations[i][j] = torch.tensor(self.env.observation_space['observation'].sample(), dtype=torch.float)

                    actions[i], _ = self.optimizer(
                        self.population2state(populations[i]), actions[i], actions_std[i], targets, value, return_std=True)
                    value = values[i] = self.value(populations[i], actions[i], targets, value)
                    print(value.mean())
                    targets = populations[i]
                    """"
                    print(values[i].mean())
                    if i == N-1:
                        idx = 0
                        for pop, v in zip(populations[i], value):
                            print(pop, v, self._reached[idx])
                            idx += 1
                            """
                    self.print_value(targets, value, f'img_{iter_}_{i}.jpg')

                cur = states[0][None, :]
                for i in range(1, N):
                    reward = self.value(cur, actions[i-1], populations[i], values[i], min_dim=0)
                    reward[reward != reward] = self.inf
                    _, topk_idx = (-reward).topk(k=self.num_elite, dim=0) # choose the minimum

                    elite = populations[i].index_select(0, topk_idx)
                    #cur = states[i][None,:]
                    cur = elite[0][None,:] # break the tie, make sure it's a trajectory... actually we'd better do sample here..

                    states[i] = states[i] * self.alpha + elite.mean(dim=0) * (1 - self.alpha)
                    states_std[i] = ((states_std[i] ** 2) * self.alpha + (elite.std(dim=0, unbiased=False) ** 2) * (1 - self.alpha)) ** 0.5

                    #cur = self._reached

                    #print(populations[i][topk_idx], reward[topk_idx], values[i][topk_idx])

                print(states)

        return states, states_std, actions, actions_std
