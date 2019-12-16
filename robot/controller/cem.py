# controller for forward dynamics
import torch
from robot.utils import batched_index_select, togpu

class CEMPlanner:
    def __init__(self, eval_function, iter_num, num_mutation, num_elite,
                 select_method='topk', std=0.2):
        self.eval_function = eval_function
        self.iter_num = iter_num
        self.num_mutation = num_mutation
        self.num_elite = num_elite
        self.select_method = select_method
        self.std = std

    def sample(self, mean, std, amounts):
        action = torch.distributions.Normal(
            loc=mean, scale=std).sample(sample_shape=(amounts,))
        dim = len(action.shape)
        action = action.permute(1, 0, *[i for i in range(2, dim)]).contiguous()
        return action

    def fit(self, actions):
        return actions.mean(dim=1), actions.std(dim=1)

    def __call__(self, scene, initial, std=None):
        # initial: batch, dim, time_step
        mean = initial
        std = self.std if std is None else std
        std = mean * 0 + std

        with torch.no_grad():
            for idx in range(self.iter_num):
                populations = self.sample(mean, std, self.num_mutation)  # sample from mean data, std data
                reward = self.eval_function(scene, populations)

                _, topk_idx = (-reward).topk(k=self.num_elite, dim=1)
                elite = batched_index_select(populations.data, 1, topk_idx)
                mean, std = self.fit(elite)
        return mean


def cem(env, state, timestep, *args, **kwargs):
    # random sample action and do optimization with CEMPlanner
    state = state[None,:]
    action = togpu([env.action_space.sample() for i in range(timestep)])[None,:]

    from .misc import evalulate
    planner = CEMPlanner(lambda state, actions: evalulate(env, state, actions), *args, **kwargs)
    return planner(state, action, std=0.2)
