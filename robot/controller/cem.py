# controller for forward dynamics
import numpy as np
import torch
import scipy.stats as stats
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
            upper_bound = torch.Tensor(upper_bound)
        if lower_bound is not None:
            lower_bound = torch.Tensor(lower_bound)

        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.inf = inf

    def __call__(self, scene, mean=None, std=None):
        shape = (self.num_mutation,) + tuple(mean.shape)
        # initial: batch, dim, time_step
        with torch.no_grad():
            if std is None:
                std = mean * 0 + self.std
            for idx in range(self.iter_num):
                _std = std
                if self.upper_bound is not None and self.upper_bound is not None:
                    lb_dist = mean - self.lower_bound.to(mean.device)
                    ub_dist = -mean + self.upper_bound.to(mean.device)
                    _std = torch.min(torch.abs(torch.min(lb_dist, ub_dist)/2), std)

                from robot.utils import Timer
                if self.trunc_norm:
                    #populations = torch.Tensor(self.sampler(size=shape)).to(mean.device) * _std[None, :] + mean[None, :]
                    populations = trunc_norm(shape, device=mean.device) * _std[None,:] + mean[None,:]
                else:
                    populations = torch.randn(shape, device=mean.device) *_std[None,:] + mean[None,:]
                reward = self.eval_function(scene, populations)
                reward[reward != reward] = self.inf

                _, topk_idx = (-reward).topk(k=self.num_elite, dim=0)
                assert populations.shape[0] > topk_idx.max() and topk_idx.min() >= 0,\
                    f"{populations.shape} {topk_idx.shape} topk max: {topk_idx.max()} min: {topk_idx.min()}, reward: {reward}, topk: {topk_idx}"

                elite = populations.index_select(0, topk_idx)

                mean = mean * self.alpha + elite.mean(dim=0) * (1 - self.alpha)
                std = std * self.alpha + elite.std(dim=0) * (1 - self.alpha)
        return mean
