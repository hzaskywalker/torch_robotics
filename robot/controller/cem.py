# controller for forward dynamics
import tqdm
import torch
from robot.utils.trunc_norm import trunc_norm


class CEM:
    def __init__(self, eval_function, iter_num, num_mutation, num_elite, std=0.2,
                 alpha=0., upper_bound=None, lower_bound=None, trunc_norm=False, inf=int(1e9)):
        self.eval_function = eval_function

        self.iter_num = iter_num
        self.num_mutation = num_mutation
        self.num_elite = num_elite
        self.std = std # initial std

        self.alpha = alpha
        self.trunc_norm = trunc_norm

        if upper_bound is not None:
            upper_bound = torch.tensor(upper_bound, dtype=torch.float)

        if lower_bound is not None:
            lower_bound = torch.tensor(lower_bound, dtype=torch.float)

        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.inf = inf

    def __call__(self, scene, mean=None, std=None, show_progress=False):
        shape = (self.num_mutation,) + tuple(mean.shape)
        # initial: batch, dim, time_step
        _x = mean

        with torch.no_grad():
            if std is None:
                std = mean * 0 + self.std
            ran = range if not show_progress else tqdm.trange
            for idx in ran(self.iter_num):
                _std = std
                if self.upper_bound is not None and self.upper_bound is not None:
                    lb_dist = mean - self.lower_bound.to(mean.device)
                    ub_dist = -mean + self.upper_bound.to(mean.device)
                    _std = torch.min(torch.abs(torch.min(lb_dist, ub_dist)/2), std)

                if self.trunc_norm:
                    noise = trunc_norm(shape, device=mean.device)
                else:
                    noise = torch.randn(shape, device=mean.device)
                populations = noise * _std[None, :] + mean[None, :]
                assert noise.shape == populations.shape

                reward = self.eval_function(scene, populations)
                reward[reward != reward] = self.inf
                print(reward)

                _, topk_idx = (-reward).topk(k=self.num_elite, dim=0)

                assert populations.shape[0] > topk_idx.max() and topk_idx.min() >= 0,\
                    f"{populations.shape} {topk_idx.shape} topk max: {topk_idx.max()} min: {topk_idx.min()}, reward: {reward}, topk: {topk_idx}"

                elite = populations.index_select(0, topk_idx)

                mean = mean * self.alpha + elite.mean(dim=0) * (1 - self.alpha)
                std = ((std ** 2) * self.alpha + (elite.std(dim=0, unbiased=False) ** 2) * (1 - self.alpha)) ** 0.5
        m = mean - _x
        return mean
