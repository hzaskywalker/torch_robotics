import gym
from gym.envs.registration import register


register(
    id='MBRLCartpole-v0',
    entry_point='robot.envs.gym.cartpole:CartpoleEnv'
)


register(
    id='MBRLPusher-v0',
    entry_point='robot.envs.gym.pusher:PusherEnv'
)


register(
    id='MBRLReacher3D-v0',
    entry_point='robot.envs.gym.reacher:Reacher3DEnv'
)


register(
    id='MBRLHalfCheetah-v0',
    entry_point='robot.envs.gym.half_cheetah:HalfCheetahEnv'
)


register(
    id='MBRLHalfCheetah-v1',
    entry_point='robot.envs.gym.half_cheetah:HalfCheetahEnv2'
)


def make(env_name):
    env = gym.make(env_name)
    #env.state_prior = PRIORS[env_name]
    return env
