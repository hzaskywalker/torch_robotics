from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from ..extension import ExtensionBase
import torch


class HalfCheetahPrior(ExtensionBase):
    def __init__(self):
        super(HalfCheetahPrior, self).__init__()
        self.observation_shape = (18,)
        self.derivative_shape = (18,)
        self.TASK_HORIZON = 1000

    def encode_obs(self, obs):
        if isinstance(obs, np.ndarray):
            return np.concatenate([obs[..., 1:2], np.sin(obs[..., 2:3]), np.cos(obs[..., 2:3]), obs[..., 3:]], axis=-1)
        elif isinstance(obs, torch.Tensor):
            return torch.cat([
                obs[..., 1:2],
                obs[..., 2:3].sin(),
                obs[..., 2:3].cos(),
                obs[..., 3:]
            ], dim=-1)

    def add(self, obs, pred):
        assert isinstance(obs, torch.Tensor)
        return torch.cat([
            pred[..., :1],
            obs[..., 1:] + pred[..., 1:]
        ], dim=-1)

    def obs_cost_fn(self, obs):
        return -obs[..., 0]

    def ac_cost_fn(self, acs):
        return 0.1 * (acs ** 2).sum(dim=-1)

    def cost(self, s, a, t, it=None):
        return self.obs_cost_fn(s) + self.ac_cost_fn(a)


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.prev_qpos = None
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/half_cheetah.xml' % dir_path, 5)
        utils.EzPickle.__init__(self)
        self.extension = HalfCheetahPrior()

    def step(self, action):
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()

        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = ob[0] - 0.0 * np.square(ob[2])
        reward = reward_run + reward_ctrl

        done = False
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            (self.sim.data.qpos.flat[:1] - self.prev_qpos[:1]) / self.dt,
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + np.random.normal(loc=0, scale=0.001, size=self.model.nq)
        qvel = self.init_qvel + np.random.normal(loc=0, scale=0.001, size=self.model.nv)
        self.set_state(qpos, qvel)
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.25
        self.viewer.cam.elevation = -55

class GoalHalfCheetahEnv(HalfCheetahEnv):
    def __init__(self):
        super(GoalHalfCheetahEnv, self).__init__()

        from gym.spaces import Dict, Box
        self.observation_space = Dict(
            observation=self.observation_space,
            desired_goal=Box(low=-1, high=1, shape=(1,)),
            achieved_goal = Box(low=-1, high=1, shape=(1,))
        )

    def _get_obs(self):
        return {
            'observation': super(GoalHalfCheetahEnv, self)._get_obs(),
            'achieved_goal': np.zeros(1),
            'desired_goal': np.zeros(1),
        }

    def step(self, action):
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()

        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = ob['observation'][0] - 0.0 * np.square(ob['observation'][2])
        reward = reward_run + reward_ctrl

        done = False
        return ob, reward, done, {}

class HalfCheetahEnv2(HalfCheetahEnv):
    def __init__(self):
        self.prev_qpos = None
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/half_cheetah.xml' % dir_path, 1)
        utils.EzPickle.__init__(self)
