# Wrapper of the model and model based controller
import numpy as np
import gym
import os
import tqdm
from robot.utils import rollout, AgentBase, tocpu, evaluate
from robot.utils.trainer import merge_training_output
from .replay_buffer import ReplayBuffer
from .planner import RolloutCEM
from .model import EnBNNAgent


class Worker:
    """
    not end2end model-based controller
    contains the following part:
        model_agent:
            1. network and its optimizer
            2. a framebuffer where we can use it to udpate the model
        controller:
            1. take model as a parameter and output the actions
    """
    def __init__(self, env, model, maxlen=int(1e6), timestep=100,
                 num_train=50, batch_size=200,
                 horizon=20, num_mutation=500, num_elite = 50, recorder=None):
        assert isinstance(model, AgentBase)
        self.env = env
        self.model: EnBNNAgent = model
        self.controller = RolloutCEM(self.model, self.env.action_space, horizon=horizon, num_mutation=num_mutation, num_elite=num_elite)

        self.buffer = ReplayBuffer(maxlen, timestep)
        self.num_train = num_train
        self.batch_size = batch_size

        self.num_train_step = num_train
        self.timestep = timestep
        self.recoder = recorder

    def __call__(self, x, goal):
        out = tocpu(self.controller(x, goal))
        return out

    def epoch(self, n_rollout):
        # new rollout
        mb_obs, mb_actions = [], []
        total_reward = 0
        for _ in range(n_rollout):

            observation = self.env.reset()
            obs, ag, g = [observation[i] for i in ['observation', 'achieved_goal', 'desired_goal']]
            ep_obs, ep_actions = [], []
            for t in range(self.timestep):
                action = self(obs, g)

                observation_new, r, done, info = self.env.step(action)
                total_reward += r
                obs_new = observation_new['observation']
                ep_obs.append(obs.copy()); ep_actions.append(action.copy())
                obs = obs_new

                if done:
                    break
            ep_obs.append(obs.copy())
            mb_obs.append(ep_obs); mb_actions.append(ep_actions)
        total_reward /= n_rollout

        # store data
        batch = [np.array(i) for i in [mb_obs, mb_actions]]
        # store the episodes
        self.buffer.store_episode(batch)

        # update normalizer
        self.model.update_normalizer(batch[0], 'obs')
        self.model.update_normalizer(batch[1], 'action')

        # train networks
        tmp = self.buffer.get()
        s, a, t = tmp['obs'][...,:-1, :], tmp['actions'], tmp['obs'][...,1:, :]
        s = s.reshape(-1, s.shape[-1])
        a = a.reshape(-1, a.shape[-1])
        t = t.reshape(-1, t.shape[-1])
        idxs = np.arange(len(s))

        for i in range(self.num_train):
            idxs = np.random.permutation(idxs)

            batch_size = self.batch_size
            num_batch = (len(idxs) + batch_size - 1) // batch_size

            for j in range(num_batch):
                idx = idxs[j * batch_size:(j + 1) * batch_size]
                self.model.update(s[idx], a[idx], t[idx])
        return total_reward

