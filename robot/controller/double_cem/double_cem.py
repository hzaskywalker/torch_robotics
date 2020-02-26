# controller for forward dynamics
import tqdm
import torch
from robot.utils.trunc_norm import trunc_norm

class CEM:
    def __init__(self, eval_function, iter_num, num_mutation, num_elite,
                 alpha=0., trunc_norm=False, inf=int(1e9),
                 action_std = 0.2, state_std = None,
                 s_upper_bound = None, s_lower_bound = None,
                 ac_upper_bound=None, ac_lower_bound=None):

        """
        In fact, there should be multi-variants..
        For example, we don't have to optimize the state per step, we can consider them as some meta things..
        And we can sample multi trajectory for each states.., but I don't know if this is useful..
            - Suppose we sampled 10 states, and for each states we sample 10 to evaluate the reward of these 10 states:
                we will use 10 elites of 100 of return as the score of each state and then we select the top 10 states at the current timestep
                this is not the same as select top 10 from different states and actions together
                which is better? It's hard to answer the questions.. the first one seems can do exploration better..
        """

        self.eval_function = eval_function

        self.iter_num = iter_num
        self.num_mutation = num_mutation
        self.num_elite = num_elite
        self.action_std = action_std # initial std
        self.state_std = state_std if state_std is not None else self.action_std

        self.alpha = alpha
        self.trunc_norm = trunc_norm

        if s_upper_bound is not None:
            s_upper_bound = torch.tensor(s_upper_bound, dtype=torch.float)
        if s_lower_bound is not None:
            s_lower_bound = torch.tensor(s_lower_bound, dtype=torch.float)
        self.s_upper_bound = s_upper_bound
        self.s_lower_bound = s_lower_bound

        if ac_upper_bound is not None:
            ac_upper_bound = torch.tensor(ac_upper_bound, dtype=torch.float)
        if ac_lower_bound is not None:
            ac_lower_bound = torch.tensor(ac_lower_bound, dtype=torch.float)
        self.ac_upper_bound = ac_upper_bound
        self.ac_lower_bound = ac_lower_bound

        self.inf = inf

    def sample(self, mean, std, lb, ub, trunc_norm):
        shape = (self.num_mutation,) + tuple(mean.shape)
        _std = std
        if lb is not None and ub is not None:
            lb_dist = mean - lb.to(mean.device)
            ub_dist = -mean + ub.to(mean.device)
            _std = torch.min(torch.abs(torch.min(lb_dist, ub_dist) / 2), std)

        if self.trunc_norm:
            noise = trunc_norm(shape, device=mean.device)
        else:
            noise = torch.randn(shape, device=mean.device)
        populations = noise * _std[None, :] + mean[None, :]
        assert noise.shape == populations.shape
        return populations

    def __call__(self, scene, state, action, state_std=None, action_std=None, show_progress=False):
        # scene (d,)
        # state_mean (N, state_dim) N subgoals
        # action_mean (N, K, action_dim) N subgoals, K for each steps..

        with torch.no_grad():
            if state_std is None:
                state_std = state * 0 + self.state_std
            if action_std is None:
                action_std = state * 0 + self.action_std
            ran = range if not show_progress else tqdm.trange
            for _ in ran(self.iter_num):
                state_population = self.sample(state, state_std, self.s_lower_bound, self.s_upper_bound, self.trunc_norm)
                action_population = self.sample(action, action_std, self.ac_lower_bound, self.ac_upper_bound, self.trunc_norm)

                reward = self.eval_function(scene, state_population, action_population)
                reward[reward != reward] = self.inf

                # reward shape (T:)

                _, topk_idx = (-reward).topk(k=self.num_elite, dim=0)
                elite = populations.index_select(0, topk_idx)

                mean = mean * self.alpha + elite.mean(dim=0) * (1 - self.alpha)
                std = ((std ** 2) * self.alpha + (elite.std(dim=0, unbiased=False) ** 2) * (1 - self.alpha)) ** 0.5
        m = mean - _x
        return mean
