import gym
import numpy as np
from robot.controller.her.ddpg_agent import DDPGAgent
from robot.utils.rl_utils import RLRecorder
import gym

class Wrapper(gym.Wrapper):
    def __init__(self, env):
        super(Wrapper, self).__init__(env)
        self.observation_space = gym.spaces.Dict(
            observation=gym.spaces.Box(-np.inf, np.inf, (21+3,)),
            desired_goal =env.observation_space['desired_goal'],
            achieved_goal=env.observation_space['achieved_goal'],
        )

    def wrap_obs(self, obs):
        s = obs['observation']
        obs['observation'] = np.concatenate([np.cos(s[:7]), np.sin(s[:7]), s[7:]])
        return obs

    def step(self, action):
        obs, r, d, info = self.env.step(action)
        return self.wrap_obs(obs), r, d, info

    def reset(self):
        return self.wrap_obs(self.env.reset())


def make(env_name):
    from robot import A
    env = Wrapper(A.train_utils.make(env_name))
    return env

def Arm():
    timestep = 100
    n_batch = 40
    env_name = 'arm'

    recorder = RLRecorder(env_name, 'arm_her', save_model=slice(100000000, None, 1), network_loss=slice(0, None, 50),
                          evaluate=slice(0, None, 50), save_video=1, max_timestep=timestep, make=make)

    DDPGAgent(
        n=8, num_epoch=25000, timestep=timestep, n_rollout=2, n_batch=n_batch,
        make=make, env_name=env_name, noise_eps=0.2, random_eps=0.3,
        batch_size=256, future_K=4,
        gamma=0.98, lr=0.001, tau=0.05, update_target_period=1, clip_critic=False, # no clip
        device='cuda:0', recorder=recorder,
    ) # 50

if __name__== '__main__':
    Arm()
