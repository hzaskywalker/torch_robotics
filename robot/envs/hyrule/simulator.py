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

PxIdentity = np.array([1, 0, 0, 0])
x2y = np.array([0.7071068, 0, 0, 0.7071068])
x2z = np.array([0.7071068, 0, 0.7071068, 0])


def add_link(builder, father, link_pose, local_pose=None, name=None, joint_name=None, range=None,
             friction=0., damping=0., stiffness=0., type='hinge', father_pose_type='mujoco', contype=1, conaffinity=1):
    # range  [a, b]
    link = builder.create_link_builder(father)
    link.set_name(name)

    if father is not None:
        assert type in ['hinge', 'slider']
        link_pose = np.array(link_pose[0]), np.array(link_pose[1])
        local_pose = np.array(local_pose[0]), np.array(local_pose[1])

        def parent_pose(xpos, xquat, ypos, yquat):
            pos = rotate_vector(ypos, xquat) + xpos
            quat = qmult(xquat, yquat)
            return Pose(pos, quat)

        if type == 'hinge':
            joint_type = sapien_core.ArticulationJointType.REVOLUTE
        else:
            joint_type = sapien_core.ArticulationJointType.PRISMATIC

        link.set_joint_name(joint_name)
        father_pose = parent_pose(*link_pose, *local_pose) if father_pose_type == 'mujoco' else Pose(*link_pose)
        link.set_joint_properties(
            joint_type, np.array([range]),
            father_pose, Pose(*local_pose),
            friction, damping
        )
        link.set_collision_group(contype, conaffinity, 0)
    return link


class Simulator:
    """
    Major interface...
    """
    def __init__(self, timestep=0.01, gravity=(0, 0, -9.8)):
        self.timestep = timestep
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.sim = sapien_core.Simulation()
        self._optifuser = sapien_core.OptifuserRenderer()
        self.sim.set_renderer(self._optifuser)
        self.scene = self.sim.create_scene(gravity=np.array(gravity))
        self.scene.set_timestep(timestep)

        self.seed()
        self.agent = None # agent is always special in the scene, it should be the only articulation object
        self.build_scene()


    @property
    def dt(self):
        return self.timestep

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


    def do_simulation(self):
        self.scene.step()

    def build_scene(self):
        raise NotImplementedError

    def build_renderer(self):
        raise NotImplementedError

    def step(self, action=None):
        self.do_simulation(self.frame_skip)

    def __del__(self):
        self.sim = None
        self.scene = None


