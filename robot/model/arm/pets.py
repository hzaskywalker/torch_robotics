# reproduce the pets cem with the new framework
import os
import torch
import tqdm
from torch import nn
import numpy as np
from robot.controller.pets.model import EnBNN

from robot import U
from . import Frame, trainer

class Distribution(Frame):
    # it always maintain dim-2 as the ensemble dimension..
    def __init__(self, state, mean=None, log_var=None):
        self.state = state
        self.mean = mean
        self.log_var = log_var
        if state is None:
            assert mean is not None and log_var is not None
            self.state = self.sample(mean, log_var)

    def iter(self):
        return (self.state, self.mean, self.log_var)

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
        loss = loss.mean(dim=(0, 1, 3))
        loss = loss.sum() # sum across different models
        return {'loss': loss}

    @classmethod
    def from_observation(cls, observation):
        # build new frame from the observation
        return cls(observation[..., None, :], None, None) # notice, there is no mean, std, but only state

    def as_observation(self):
        return U.tocpu(self.state)


class CheetahFrame(Distribution):
    input_dims = (5, 18, 6)
    output_dims = (18, 18)

    def as_input(self, action):
        s = self.state
        assert s.dim() == 3
        inp = torch.cat([s[..., 1:2], s[..., 2:3].sin(), s[..., 2:3].cos(), s[..., 3:]], dim=-1)
        return inp, action

    def add(self, mean, std):
        mean =  torch.cat([mean[..., :1], self.state[..., 1:] + mean[..., 1:]], dim=-1)
        return self.__class__(None, mean, std)

    @classmethod
    def obs_cost_fn(cls, obs):
        return -obs[..., 0]

    @classmethod
    def ac_cost_fn(cls, acs):
        return 0.1 * (acs ** 2).sum(dim=-1)

    def compute_reward(self, s, a, goal):
        return -(self.obs_cost_fn(s.state) + self.ac_cost_fn(a))


class ArmFrame(Distribution):
    input_dims = (5, 21, 7)
    output_dims = (14 + 3, 14)

    def as_input(self, action):
        s = self.state
        q, dq = s[..., :7], s[..., 7:14]
        inp = torch.cat([q.sin(), q.cos(), dq], dim=-1)
        return inp, action

    def add(self, mean, std):
        # maybe I need to bound the rotations..?
        mean = torch.cat([mean[...,:14]+self.state[...,:14], mean[...,14:17]], dim=-1)
        return self.__class__(None, mean, std)

    def compute_reward(self, s, a, g):
        while g.dim() < self.state.dim():
            g = g[None,:]
        return -((self.state[...,14:17] - g)**2).sum(dim=-1) ** 0.5



class PlaneFrame(Distribution):
    input_dims = (5, 2, 2)
    output_dims = (2, 2)

    def as_input(self, action):
        return self.state, action

    def add(self, mean, std):
        return self.__class__(None, self.state + mean, std)

    def compute_reward(self, s, a, g):
        while g.dim() < self.state.dim():
            g = g[None,:]
        return -(((self.state - g) ** 2).sum(dim=-1)) ** 0.5


class EnsembleModel(EnBNN):
    def __init__(self, inp_dims, oup_dims,
                 mid_channels = 200,
                 num_layers=5,
                 weight_decay = np.array([0.000025, 0.00005, 0.000075, 0.000075, 0.0001]),
                 var_reg=0.01):

        self._weight_decay = weight_decay
        self._var_reg = var_reg
        ensemble_size, state_dim, action_dim = inp_dims
        super(EnsembleModel, self).__init__(ensemble_size, state_dim + action_dim, oup_dims[0],
                                            mid_channels=mid_channels, num_layers=num_layers)

    def forward(self, obs, action):
        obs = obs.transpose(1, 0)
        if action.dim() == 2:
            action = action[:, None] # add batch dimension
        action = action.transpose(1, 0)
        return [i.transpose(1, 0) for i in super(EnsembleModel, self).forward(obs, action)]

    def loss(self):
        return self._var_reg * self.var_reg() + self.decay(self._weight_decay)


class Dataset:
    def __init__(self, max_timestep, size, batch_size, frame_type):
        from robot.controller.pets.replay_buffer import ReplayBuffer
        self.timestep = max_timestep
        self.dataset = ReplayBuffer(self.timestep, size)
        self.batch_size = batch_size
        self.frame_type = frame_type

    def store_episode(self, data):
        self.dataset.store_episode(data)

    def gen_data(self, num_train):
        tmp = self.dataset.get()
        s, a, t = tmp['obs'][..., :-1, :], tmp['actions'], tmp['obs'][..., 1:, :]
        s = s.reshape(-1, s.shape[-1])
        a = a.reshape(-1, a.shape[-1])
        t = t.reshape(-1, t.shape[-1])
        idxs = np.arange(len(s))

        for _ in tqdm.trange(num_train):
            idxs = np.random.permutation(idxs)

            batch_size = self.batch_size
            num_batch = (len(idxs) + batch_size - 1) // batch_size

            for j in tqdm.trange(num_batch):
                idx = idxs[j * batch_size:(j + 1) * batch_size]

                state = self.frame_type.from_observation(s[idx])
                future = self.frame_type.from_observation(t[idx][:, None])
                yield state.cuda(), U.togpu(a[idx][:, None]), future.cuda()

class PetsRollout:
    def __init__(self, model, frame_type, npart=20):
        self.model = model
        self.cls = frame_type
        self.ensemble_size = self.model.model.ensemble_size
        self.npart = npart
        self.K = self.npart//self.ensemble_size

    def rollout(self, obs, a, goal):
        obs = U.togpu(obs)
        a = U.togpu(a)
        if goal is not None:
            goal = U.togpu(goal)

        s = self.cls.from_observation(obs).state

        s = s.expand(-1, self.npart, -1).reshape(-1, self.ensemble_size, *s.shape[2:])
        s = self.cls(s)

        a = a[:, :, None, None, :].expand(-1, -1, self.K, self.ensemble_size, a.shape[-1]) # b, time, sene
        a = a.transpose(2, 1).reshape(-1, a.shape[1], self.ensemble_size, a.shape[-1])

        predict, reward = self.model.rollout(s, a, goal)
        reward = reward.reshape(-1, self.K, self.ensemble_size)
        reward = reward.mean(dim=(1, 2))
        return None, -reward


class online_trainer(trainer):
    def set_env(self):
        from robot.controller.pets.envs import make
        self.env, _ = make(self.args.env_name)
        if self.args.env_name == 'cheetah':
            self.frame_type = CheetahFrame
            timestep = 1000
        elif self.args.env_name == 'plane':
            self.frame_type = PlaneFrame
            timestep = 50
        elif self.args.env_name == 'arm':
            self.frame_type = ArmFrame
            timestep = 100
        else:
            raise NotImplementedError

        self.dataset = Dataset(timestep, int(1e6), 32, self.frame_type)

    def set_model(self):
        self.model = EnsembleModel(self.frame_type.input_dims, self.frame_type.output_dims)

    def set_agent(self):
        from .agents.simple_rollout import RolloutAgent
        normalizer = nn.ModuleList([U.Normalizer((i,)) for i in self.frame_type.input_dims[1:]])
        self.agent = RolloutAgent(self.model, lr=self.args.lr, loss_weights={'model_decay': 1.,'loss': 1.}, normalizers=normalizer).cuda()

    def set_rollout_model(self):
        self.rollout_predictor = PetsRollout(self.agent, self.frame_type, npart=20)

    def set_policy(self):
        self.set_rollout_model()
        args = self.args
        from .train_utils import RolloutCEM
        env = self.env
        self.controller = RolloutCEM(self.rollout_predictor, env.action_space, iter_num=5, horizon=30,
                                     num_mutation=500, num_elite=50, device=args.device, alpha=0.1, trunc_norm=True,
                                     lower_bound=env.action_space.low, upper_bound=env.action_space.high)

    def epoch(self, num_train=5, num_valid=0, num_eval=0, use_tqdm=False):
        print(f"########################EPOCH {self.epoch_num}###########################")
        # update buffer by run once
        self.agent.eval()
        if self.epoch_num == 0:
            policy = lambda x: self.env.action_space.sample()
        else:
            policy = self.controller
        env = self.env
        avg_reward, trajectories = U.eval_policy(policy, env, eval_episodes=1, save_video=0, progress_episode=True,
                                     video_path=os.path.join(self.path, "video{}.avi"), return_trajectories=True,
                                     timestep=self.dataset.timestep)

        for i in trajectories:
            obs = np.array([i['observation'] for i in i[0]])[None, :]
            action = np.array(i[1])[None, :]
            self.dataset.store_episode([obs, action])

            obs_inp  = obs[:, :-1].reshape(-1, obs.shape[-1])
            action_inp  = action.reshape(-1, action.shape[-1])
            obs_inp, action_inp = self.frame_type.from_observation(U.togpu(obs_inp)).as_input(U.togpu(action_inp))
            self.agent.update_normalizer([U.togpu(obs_inp), U.togpu(action_inp)])

        # train with the dataset
        self.agent.train()

        train_output = []
        for data in self.dataset.gen_data(num_train):
            out = self.agent.update(*data)
            train_output.append(out)

        out = U.merge_training_output(train_output)
        out['reward'] = avg_reward
        self.vis(out)
