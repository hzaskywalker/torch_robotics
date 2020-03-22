# make
import gym
from gym import utils
from gym.spaces import Dict, Box
from .engine import Articulation2D
from .rendering import Viewer
from .cv2_rendering import cv2Viewer
import numpy as np

class GoalAcrobat(gym.Env, utils.EzPickle):
    def __init__(self):
        gym.Env.__init__(self)
        utils.EzPickle.__init__(self)

        self.build_model()

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
