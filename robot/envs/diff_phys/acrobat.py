# make
import torch
import gym
from gym import utils
from gym.spaces import Dict, Box
from .engine import Articulation2D
from .rendering import Viewer
from .cv2_rendering import cv2Viewer
import numpy as np

class GoalAcrobat(gym.Env, utils.EzPickle):
    def __init__(self, reward_type='dense', eps=0.1):
        gym.Env.__init__(self)
        utils.EzPickle.__init__(self)

        self.eps = eps
        self.total_length = 2
        self.reward_type = reward_type

        goal_space = Box(low=np.array([-self.total_length, -self.total_length]),
                         high=np.array([self.total_length, self.total_length]))

        self.observation_space = Dict({
            'observation': Box(low=-np.inf, high=np.inf, shape=(6,)),
            'desired_goal': goal_space,
            'achieved_goal': goal_space
        })
        self.action_space = Box(low=-1, high=1, shape=(2,))
        self.action_range = 10


        self.build_model()
        self.init_qpos = self.articulator.get_qpos().detach().cpu().numpy()
        self.init_qvel = self.articulator.get_qvel().detach().cpu().numpy()

        self.viewer = None
        self._viewers = {}

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                #self.viewer = mujoco_py.MjViewer(self.sim)
                self.viewer = Viewer(500, 500)
            elif mode == 'rgb_array':
                self.viewer = cv2Viewer(500, 500)

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer


    def viewer_setup(self):
        bound = 2.2  # 2.2 for default
        self.viewer.set_bounds(-bound, bound, -bound, bound)

    def render(self, mode='human'):
        viewer = self._get_viewer(mode)
        viewer.draw_line((-2.2, 1), (2.2, 1))
        self.articulator.draw_objects(viewer)
        return viewer.render(return_rgb_array = mode=='rgb_array')


    def build_model(self):
        articulator = Articulation2D(timestep=0.2)

        M01 = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, -0.5],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        # v = -w x q
        # we choose q = [0, -0.5, 0] =>
        w, q = [0, 0, 1], [0, 0.5, 0]
        screw1 = w + (-np.cross(w, q)).tolist()
        # m = 1
        G1 = np.diag([1, 1, 1, 1, 1, 1])
        link1 = articulator.add_link(M01, screw1)
        link1.set_inertial(np.array(G1))
        link1.add_box_visual([0, 0, 0], [0.1, 0.5, 0], (0, 0.8, 0.8))
        link1.add_circle_visual((0, 0.5, 0), 0.1, (0.8, 0.8, 0.))

        M12 = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, -1.0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        screw2 = screw1
        # m = 1
        G2 = np.diag([1, 1, 1, 1, 1, 1])
        link2 = articulator.add_link(M12, screw2)
        link2.set_inertial(np.array(G2))
        link2.add_box_visual([0, 0, 0], [0.1, 0.5, 0], (0, 0.8, 0.8))
        link2.add_circle_visual((0, 0.5, 0), 0.1, (0.8, 0.8, 0.))

        EE = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, -0.5],
                [0, 0, 1, 0.],
                [0, 0, 0, 1.],
            ]
        )
        articulator.set_ee(EE)
        articulator.build()

        self.articulator = articulator


    def set_state(self, qpos, qvel):
        self.articulator.set_qpos(qpos)
        self.articulator.set_qvel(qvel)


    def reset(self):
        qpos = self.init_qpos + np.random.uniform(low=-0.01, high=0.01, size=np.shape(self.init_qpos))
        qvel = self.init_qvel + np.random.uniform(low=-0.01, high=0.01, size=np.shape(self.init_qvel))
        self.set_state(qpos, qvel)

        self._timestep = 0
        #self._goal = self.observation_space['desired_goal'].sample()

        r = (np.random.random() * 0.6+0.4) * self.total_length
        theta = np.random.random() * np.pi * 2 - np.pi
        self._goal = np.array((np.sin(theta) * r, np.cos(theta) * r))
        return self._get_obs()

    def _get_obs(self):
        q = self.articulator.get_qpos().detach().cpu().numpy()
        qvel = self.articulator.get_qvel().detach().cpu().numpy()
        achieved_goal = self.articulator.forward_kinematics()[-1][:2, 3].detach().cpu().numpy()

        return {
            'observation': np.concatenate([q, qvel, achieved_goal]),
            'desired_goal': np.array(self._goal).copy(),
            'achieved_goal': achieved_goal.copy(),
        }

    def step(self, action):
        action = action.clip(-1, 1)
        self.articulator.set_qf(action * self.action_range)
        with torch.no_grad():
            self.articulator.step()

        ob = self._get_obs()
        reward = self.compute_reward(ob['achieved_goal'], ob['desired_goal'])
        if self.reward_type == 'dense':
            is_success = (-reward < self.eps)
        else:
            is_success = reward > -0.5

        done = False
        return ob, reward, done, {'is_success': is_success}


    def compute_reward(self, achieved_goal, desired_goal, info=None):
        # TODO: hack now, I don't want to implement a pickable reward system...
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        if self.reward_type == 'dense':
            return -d
        else:
            return -(d > self.eps).astype(np.float32)
