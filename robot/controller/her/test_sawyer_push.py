import gym
import numpy as np
from robot.controller.her.ddpg_agent import DDPGAgent
from robot.envs.sapien.exp.utils import RLRecorder
from gym.wrappers import TimeLimit

class Wrapper(gym.Wrapper):
    def __init__(self, env):
        super(Wrapper, self).__init__(env)
        self.viewer = None


    def get_viewer(self):
        if self.viewer is None:
            tmp = self.env.viewer
            import mujoco_py
            self.env.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)
            self.env.viewer_setup()

            self.viewer = self.env.viewer
            self.env.viewer = tmp

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        r = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
        return obs, r, done, info

    def render(self, mode='human', **kwargs):
        self.get_viewer()
        if mode == 'rgb_array':
            self.viewer.render(500, 500)

            data = self.viewer.read_pixels(500, 500, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :]
        else:
            return self.env.render(mode=mode, **kwargs)

    def compute_reward(self, achieved_goal, desired_goal, info):
        r = np.linalg.norm(
            achieved_goal - desired_goal, axis=-1
        )
        return - (r > 0.06).astype(np.float32)


def main():
    #env = gym.make()
    make =gym.make
    timestep = 100
    n_batch = 40

    env_name = 'SawyerPushAndReachArenaTrainEnvBig-v0'

    def make2(env_name):
        from multiworld.envs import mujoco

        env = TimeLimit(Wrapper(make(env_name)), timestep)
        return env
    env = make2(env_name)
    """
    while True:
        img = env.render(mode='rgb_array')
        import cv2
        cv2.imshow('x', img)
        cv2.waitKey(0)

        a = env.action_space.sample()
        env.step(a)
    exit(0)
    """
    recorder = RLRecorder(env, '/tmp/tmp/swayerpush', save_model=slice(100000000, None, 1), network_loss=slice(0, None, 50),
                          evaluate=slice(0, None, 50), save_video=1, max_timestep=timestep)

    DDPGAgent(
        n=16, num_epoch=25000, timestep=timestep, n_rollout=2, n_batch=n_batch,
        make=make2, env_name=env_name, noise_eps=0.2, random_eps=0.3,
        batch_size=256, future_K=4,
        gamma=0.99, lr=0.001, tau=0.05, update_target_period=1, clip_critic=True,
        device='cuda:0', recorder=recorder,
    ) # 50

if __name__== '__main__':
    main()
