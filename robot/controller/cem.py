# controller for forward dynamics
import gym
import torch
from ..utils import batched_index_select, togpu
from .misc import evaluate


class CEMController:
    def __init__(self, iter_num, num_mutation, num_elite,
                 eval_function=None, env:gym.Env=None,
                 select_method='topk', std=0.2):
        self.iter_num = iter_num
        self.num_mutation = num_mutation
        self.num_elite = num_elite
        self.select_method = select_method
        self.std = std

        self.env = env
        self.eval_function = eval_function
        if eval_function is None:
            self.eval_function = lambda state, actions: evaluate(env, state, actions)


    def sample(self, mean, std, amounts):
        action = torch.distributions.Normal(
            loc=mean, scale=std).sample(sample_shape=(amounts,))
        return action

    def fit(self, actions):
        return actions.mean(dim=0), actions.std(dim=0)

    def __call__(self, scene, mean=None, horizon=None):
        # initial: batch, dim, time_step
        if mean is None:
            assert horizon is not None and self.env is not None
            mean = togpu([self.env.action_space.sample() for i in range(horizon)])

        with torch.no_grad():
            std = mean * 0 + self.std
            for idx in range(self.iter_num):
                populations: torch.Tensor = self.sample(mean, std, self.num_mutation)  # sample from mean data, std data
                reward = self.eval_function(scene, populations)
                _, topk_idx = (-reward).topk(k=self.num_elite, dim=0)
                elite = populations.index_select(0, topk_idx)
                mean, std = self.fit(elite)
        return mean
