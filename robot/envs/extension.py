class ExtensionBase:
    observation_shape = None
    action_shape = None

    def add(self, obs, delta):
        # change the delta
        return obs + delta

    def encode_obs(self, obs):
        # change the observation
        return obs

    def encode_action(self, action):
        return action

    def distance(self, obs, other, is_batch=True):
        # distance in pytorch
        assert is_batch
        obs = obs.reshape(obs.shape[0], -1)
        other = other.reshape(other.shape[0], -1)
        return ((other - obs) ** 2).sum(dim=1)

    def cost(self, s, a, t, it=None):
        raise NotImplementedError
