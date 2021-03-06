import threading
import numpy as np

"""
the replay buffer here is basically from the openai baselines code
"""


class ReplayBuffer:
    def __init__(self, env, max_timesteps, buffer_size, sample_func):
        observation_space = env.observation_space
        action_space = env.action_space
        self.T = max_timesteps
        self.size = buffer_size // self.T
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        # create the buffer to store info
        self.buffers = {'obs': np.empty([self.size, self.T + 1, observation_space['observation'].shape[0]]),
                        'ag': np.empty([self.size, self.T + 1, observation_space['desired_goal'].shape[0]]),
                        'g': np.empty([self.size, self.T, observation_space['achieved_goal'].shape[0]]),
                        'actions': np.empty([self.size, self.T, action_space.shape[0]]),
                        }

    # store the episode
    def store_episode(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        batch_size = mb_obs.shape[0]

        idxs = self._get_storage_idx(inc=batch_size)
        # store the informations
        self.buffers['obs'][idxs] = mb_obs
        self.buffers['ag'][idxs] = mb_ag
        self.buffers['g'][idxs] = mb_g
        self.buffers['actions'][idxs] = mb_actions
        self.n_transitions_stored += self.T * batch_size

    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}

        for key in self.buffers.keys():
            temp_buffers[key] = self.buffers[key][:self.current_size]

        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx