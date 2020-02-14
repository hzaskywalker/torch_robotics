import numpy as np
from gym.utils import seeding
import gym
from gym import spaces
from transforms3d.quaternions import qmult, rotate_vector, axangle2quat
from robot.envs.sapien.camera import CameraRender

DEFAULT_SIZE = 500

import sapien.core as sapien_core
print('USE sapien core')
from sapien.core import Pose


class Simulator:
    """
    Major interface...
    """
    def __init__(self, frame_skip, timestep=0.01, gravity=[0, 0, -9.8]):
        self.frame_skip = frame_skip
        self.timestep = timestep
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.sim = sapien_core.Simulation()
        self._optifuser = sapien_core.OptifuserRenderer()
        self.sim.set_renderer(self._renderer)
        self.scene = self.sim.create_scene(gravity=np.array(gravity))
        self.scene.set_timestep(timestep)

        self.builder = self.scene.create_articulation_builder()
        self.seed()
        self.build_scene()

    @property
    def dt(self):
        return self.timestep * self.frame_skip

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        self.build_renderer()

    def _get_viewer(self, mode):
        self._renderer = self._viewers.get(mode)
        if self._renderer is None:
            if mode == 'human':
                self._renderer = sapien_core.OptifuserController(self._optifuser)
            elif mode == 'rgb_array':
                self._renderer = CameraRender(self.scene, mode, width=500, height=500)

            self.viewer_setup()
            if mode == 'human':
                self._renderer.show_window()

            self._renderer.set_current_scene(self.scene)
            self._viewers[mode] = self._renderer

        self.scene.update_render()
        return self._renderer

    def render(self, mode='human'):
        return self._get_viewer(mode).render()


    def do_simulation(self, n_frames):
        for _ in range(n_frames):
            self.scene.step()

    def build_scene(self):
        raise NotImplementedError

    def build_renderer(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError



    def __del__(self):
        self.sim = None
        self.scene = None
