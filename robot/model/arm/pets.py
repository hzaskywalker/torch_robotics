# reproduce the pets cem with the new framework
import os
import torch
from torch import nn
import numpy as np
from robot import U
from robot.controller.pets.model import EnBNN

from robot import A, U

class Distribution(A.Frame):
    # it always maintain dim-2 as the ensemble dimension..
    def __init__(self, state, mean=None, log_var=None):
        self.state = state
        self.mean = mean
        self.log_var = log_var
        if state is None:
            assert mean is not None and log_var is not None
            self.state = self.sample(mean, log_var)

    def sample(self, mean, log_var):
        # sample the
        inp = mean
        if log_var is not None:
            inp += torch.randn_like(log_var) * torch.exp(log_var * 0.5)  # sample
        return inp

    def calc_loss(self, label):
        t = label.state
        inv_var = torch.exp(-self.log_var)
        loss = ((self.mean - t.detach()) ** 2) * inv_var + self.log_var
        loss = loss.mean(dim=(0, -1)).sum() # sum across different models
        return {
            'loss': loss
        }


class CheetahFrame(Distribution):
    input_dims = (5, 18, 7)
    output_dims = (18, 18)

    def as_input(self, action):
        s = self.state
        inp = torch.cat([s[..., 1:2], s[..., 2:3].sin(), s[..., 2:3].cos(), s[..., 3:]], dim=-1)
        return inp, action

    def as_observation(self):
        return U.tocpu(self.state)

    def add(self, mean, std):
        mean =  torch.cat([mean[..., :1], self.state[..., 1:] + mean[..., 1:]], dim=-1)
        return self.__class__(None, mean, std)

    @classmethod
    def from_observation(cls, observation):
        # build new frame from the observation
        return cls(observation[..., None, :], None, None) # notice, there is no mean, std, but only state

    @classmethod
    def obs_cost_fn(cls, obs):
        return -obs[..., 0]

    @classmethod
    def ac_cost_fn(cls, acs):
        return 0.1 * (acs ** 2).sum(dim=-1)

    def compute_reward(self, s, a, goal):
        return -(self.obs_cost_fn(s.state) + self.ac_cost_fn(a))


class EnsembleModel(EnBNN):
    def __init__(self, inp_dims, oup_dims,
                 mid_channels = 200,
                 num_layers=5,
                 weight_decay = np.array([0.000025, 0.00005, 0.000075, 0.000075, 0.0001]),
                 var_reg=0.01):

        self.weight_decay = weight_decay
        self.var_reg = var_reg
        ensemble_size, state_dim, action_dim = inp_dims
        super(EnsembleModel, self).__init__(ensemble_size, state_dim + action_dim, oup_dims[0],
                                            mid_channels=mid_channels, num_layers=num_layers)

    def loss(self):
        return self.var_reg * self.forward_model.var_reg() + self.forward_model.decay(self.weight_decay)


class online_trainer(A.trainer):
    def __init__(self, *args, **kwargs):
        super(online_trainer, self).__init__(*args, **kwargs)
        self.epoch_num = 0

    def get_envs(self):
        from robot.controller.pets.envs import make
        from robot.controller.pets.replay_buffer import ReplayBuffer
        self.env, _ = make('cheetah')
        self.frame_type =  CheetahFrame
        self.dataset = ReplayBuffer(1000, int(1e6))

    def get_model(self):
        self.model = EnsembleModel(self.frame_type.input_dims, self.frame_type.output_dims)

    def epoch(self, num_train, num_valid, num_eval=5, use_tqdm=False):
        # update buffer by run once
        if self.epoch_num == 0:
            policy = lambda x: self.env.action_space.sample()
        else:
            policy = self.controller
        trajectories = U.eval_policy(policy, self.env, 1, save_video=1,
                                     video_path=os.path.join(self.path, "video{}.avi"), return_trajectories=True)
        for i in trajectories:
            self.dataset.store_episode(*i)

        # update normalizer ...

        # train with the dataset
