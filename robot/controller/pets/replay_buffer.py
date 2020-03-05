import numpy as np
"""
the replay buffer here is basically from the openai baselines code
"""

class ReplayBuffer:
    def __init__(self, max_timesteps, buffer_size):
        self.T = max_timesteps
        self.size = buffer_size // self.T
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        # create the buffer to store info
        self.buffers = None

    # store the episode
    def store_episode(self, episode_batch):
        mb_obs, mb_actions = episode_batch
        batch_size = mb_obs.shape[0]

        if self.buffers is None:
            self.buffers = {
                'obs': np.empty([self.size, self.T + 1, mb_obs.shape[1]]),
                'actions': np.empty([self.size, self.T, mb_actions.shape[1]]),
            }


        idxs = self._get_storage_idx(inc=batch_size)
        # store the informations
        self.buffers['obs'][idxs] = mb_obs
        self.buffers['actions'][idxs] = mb_actions
        self.n_transitions_stored += self.T * batch_size

    # sample the data from the replay buffer
    def get(self):
        temp_buffers = {}

        for key in self.buffers.keys():
            temp_buffers[key] = self.buffers[key][:self.current_size]

        # sample transitions
        return temp_buffers

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
            raise NotImplementedError
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx