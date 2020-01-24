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
        self.inp_dim = 18
        self.oup_dim = 18
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


class HalfCheetahEnv2(HalfCheetahEnv):
    def __init__(self):
        self.prev_qpos = None
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/half_cheetah.xml' % dir_path, 1)
        utils.EzPickle.__init__(self)
