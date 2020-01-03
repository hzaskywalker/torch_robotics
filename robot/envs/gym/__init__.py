from gym.envs.registration import register


register(
    id='MBRLCartpole-v0',
    entry_point='env.cartpole:CartpoleEnv'
)


register(
    id='MBRLPusher-v0',
    entry_point='env.pusher:PusherEnv'
)


register(
    id='MBRLReacher3D-v0',
    entry_point='env.reacher:Reacher3DEnv'
)


register(
    id='MBRLHalfCheetah-v0',
    entry_point='env.half_cheetah:HalfCheetahEnv'
)


class StatePrior:
    def __init__(self, env):
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def encode(self, x):
        return x

    def delete(self, t, s):
        return t - s

    def dist(self, s, t):
        return ((t - s) ** 2).sum(dim=-1)

def make(env_name):
   pass
