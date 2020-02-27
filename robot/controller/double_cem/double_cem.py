# controller for forward dynamics
import tqdm
import torch
from robot.utils.trunc_norm import trunc_norm


class DoubleCEM:
    def __init__(self, constraint, rollout, optimizer, iter_num, num_mutation=100, num_elite=10, std=0.2,
                 alpha=0., upper_bound=None, lower_bound=None, trunc_norm=False, inf=int(1e9)):

        self.constraint = constraint # the constraint function measures the loss if two states doesn't match..
        self.rollout = rollout

        self.optimizer = optimizer

        self.iter_num = iter_num
        self.num_mutation = num_mutation
        self.num_elite = num_elite
        self.std = std
        self.alpha = alpha
        self.trunc_norm = trunc_norm
        self.inf = inf

        if upper_bound is not None:
            upper_bound = torch.tensor(upper_bound, dtype=torch.float)
        if lower_bound is not None:
            lower_bound = torch.tensor(lower_bound, dtype=torch.float)
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    def __call__(self, states, actions, states_std=None, actions_std=None, show_progress=True):
        # scene is the first states...
        # states (N, state_dim)
        # the first is the initial states

        N = len(states)
        if states_std is None:
            states_std = states * 0 + self.std
        shape = (states.shape[0], self.num_mutation, *states.shape[1:])
        values = torch.zeros((states.shape[0], self.num_mutation), device=states.device)

        with torch.no_grad():
            ran = tqdm.trange if show_progress else range
            for _ in ran(self.iter_num):
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
                for i in range(N-1, -1, -1):
                    actions[i], actions_std[i], values[i] = self.optimizer(populations[i], actions[i], actions_std[i], targets, value)
                    targets = populations[i]
                    value = values[i]

                for i in range(1, N):
                    reached, rewards = self.rollout(states[i-1], actions[i], actions_std[i])
                    assert len(reached.shape) > 1

                    reward = rewards + self.constraint(reached, populations) #reached to populations...
                    reward[reward != reward] = self.inf
                    _, topk_idx = (-reward).topk(k=self.num_elite, dim=0)

                    elite = populations[i].index_select(0, topk_idx)

                    states[i] = states[i] * self.alpha + elite.mean(dim=0) * (1 - self.alpha)
                    states_std[i] = ((states_std[i] ** 2) * self.alpha + (elite.std(dim=0, unbiased=False) ** 2) * (1 - self.alpha)) ** 0.5

        return states, states_std, actions, actions_std
