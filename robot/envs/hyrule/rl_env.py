# Goal-conditioned Environment...
import gym
import os
from gym.spaces import Box, Dict
from .simulator import Simulator
from .loader import load_scene, load_json
import numpy as np


class RLEnv(Simulator):
    def __init__(self, param_path):
        super(RLEnv, self).__init__()
        self.params = [os.path.join(param_path, i) for i in os.listdir(param_path)]
        self._param_path = None

    def reset(self, filepath=None):
        # pass
        if filepath is None:
            if self._param_path is not None:
                filepath = self._param_path
            else:
                filepath = self.params[0]

        if self._param_path != filepath:
            self._param_path = filepath
            load_scene(self, load_json(self._param_path))
            return self.reset()

        if not self._reset:
            self._start_state = self.state_vector().copy()
            obs = self._get_obs()

            self.observation_space = Dict(
                observation=Box(-np.inf, np.inf, shape=obs['observation'].shape),
                desired_goal=Box(-np.inf, np.inf, shape=obs['desired_goal'].shape),
                achieved_goal = Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape),
            )

            self.action_space = Box(-1., 1., self._actuator_joint['agent'].shape)

        else:
            self.load_state_vector(self._start_state)
            obs = self._get_obs()

        self._reset = True
        self.timestep = 0

        return obs


    def get_current_cost(self):
        cur = 0
        for cost, t in self.costs.waypoints:
            if cur + t > self.timestep:
                break
        return cost


    def _get_obs(self):
        # pass
        observations = []
        for name, item in self.objects.items():
            if name == 'agent':
                observations +=[item.get_qpos(), item.get_qvel()]
            else:
                observations +=[np.array(item.pose.p), np.array(item.pose.q), np.array(item.velocity), np.array(item.angular_velocity)]

        achieved_goal, desired_goal = self.get_current_cost()._get_obs(self)
        return {
            'observation': np.concatenate(observations),
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal
        }


    def compute_reward(self, achieved_goal, desired_goal, info=None):
        # TODO: hack now, I don't want to implement a pickable reward system...
        return -self.get_current_cost().compute_cost(achieved_goal, desired_goal, info)
