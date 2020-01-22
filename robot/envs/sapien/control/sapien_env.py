from collections import OrderedDict
import numpy as np
from gym.utils import seeding
import gym
from gym import spaces
from transforms3d.quaternions import qmult, rotate_vector, axangle2quat

DEFAULT_SIZE = 500

import sapien.core as sapien_core
print('USE sapien core')

from sapien.core import Pose


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'))
        high = np.full(observation.shape, float('inf'))
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class SapienEnv(gym.Env):
    """
    Superclass for all Sapyen environments.
    """
    def __init__(self, frame_skip, timestep=0.01):
        self.frame_skip = frame_skip
        self.timestep = timestep
        self.viewer = None
        self._viewers = {}
        self._objects = []

        self.metadata = {
            'render.modes': ['human'], 
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self._sim = sapien_core.Simulation()
        self._renderer2 = sapien_core.OptifuserRenderer()
        self._sim.set_renderer(self._renderer2)

        self.sim = self._sim.create_scene()
        self.sim.set_timestep(timestep)
        self._renderer = sapien_core.OptifuserController(self._renderer2)

        self.force_actuators = []
        self.build_render()

        self.builder = self.sim.create_articulation_builder()
        self.model, self.root = self.build_model()

        # get actors...
        joints = self.model.get_joints()
        joint_name = [i.name for i in joints]
        joint_dof = [i.get_dof() for i in joints]
        self.actor_idx = []
        self.actor_bound = []
        self._dof = len(self.model.get_qf())
        assert self._dof == sum(joint_dof)

        print(joint_name)
        print(joint_dof)
        for (name, low, high) in self.force_actuators:
            i = joint_name.index(name)
            assert joint_name[i] == name
            s = sum([joint_dof[j] for j in range(i)])
            print(name, s)
            for j in range(joint_dof[i]):
                self.actor_idx.append(s+j)
                self.actor_bound.append((low, high))
        self.actor_bound = np.array(self.actor_bound)
        self.actor_idx = np.array(self.actor_idx)
        print(self.actor_idx)
        print(self.actor_bound)

        self.init_qpos = self.get_qpos()
        self.init_qvel = self.get_qvel()
        self.seed()

        self._set_action_space()
        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done

        self._set_observation_space(observation)
        self.seed()

        self.create_window = False

    def get_qpos(self):
        if self.root is not None:
            root = [self.root.get_global_pose().p, self.root.get_global_pose().q]
        else:
            root = []
        return np.concatenate(root + [self.model.get_qpos().ravel()])

    def get_qvel(self):
        if self.root is not None:
            root = [self.root.get_linear_velocity(), self.root.get_angular_velocity()]
        else:
            root = []
        return np.concatenate(root + [self.model.get_qpos().ravel()])

    def _set_action_space(self):
        #bounds = self.model.get_force_actuator_range().copy()
        bounds = np.array(self.actor_bound)
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def build_render(self):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError


    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        if not self.create_window:
            self._renderer.show_window()
            self.create_window = True

    # -----------------------------

    def reset(self):
        return self.reset_model()


    def set_state(self, qpos, qvel):
        if self.root is not None:
            self.model.set_root_pose(qpos[:7], qvel[:6])
            qpos = qpos[7:]
            qvel = qvel[6:]
        self.model.set_qpos(qpos)
        self.model.set_qvel(qvel)


    @property
    def dt(self):
        return self.timestep * self.frame_skip

    def do_simulation(self, a, n_frames):
        qf = np.zeros((self._dof), np.float32)
        qf[self.actor_idx] = a
        for _ in range(n_frames):
            self.model.set_qf(qf)
            self.sim.step()


    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        self.viewer_setup()
        if mode == 'human':
            self.sim.update_render()
            self._renderer.render()
        else:
            raise NotImplementedError

    def close(self):
        pass

    def _get_viewer(self, mode):
        return self._renderer

    def get_body_com(self, body_name):
    #    return self.data.get_body_xpos(body_name)
        pass

    def add_capsule(self, body, xpos, xquat, radius, half_length, color, name, shape=True):
        if shape:
            body.add_capsule_shape(Pose(xpos, xquat), radius, half_length) #half length
        body.add_capsule_visual(Pose(xpos, xquat), radius, half_length, color, name) #half length

    def add_sphere(self, builder, body, xpos, xquat, radius, color, name, shape=True):
        if shape:
            builder.add_sphere_shape_to_link(body, Pose(xpos, xquat), radius) #half length
        builder.add_sphere_visual_to_link(body, Pose(xpos, xquat), radius, color, name) #half length

    def my_add_link(self, father, link_pose, local_pose=None, name=None, joint_name=None, range=None,
                    friction=0., dumping=0., type='hinge', father_pose_type='mujoco'):
        # range  [a, b]
        assert type in ['hinge', 'slider']
        link = self.builder.create_link_builder(father)
        link.set_name(name)

        if father is not None:
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
                friction, dumping
            )
        return link

    def fromto(self, link, vec, size, rgb, name, density=1000.):
        def vec2pose(vec):
            l = np.linalg.norm(vec)
            a, b = np.array([1, 0, 0]), vec/l
            # find quat such that qmult(quat, [1, 0, 0]) = vec
            if np.linalg.norm(a-b) < 1e-6:
                pose = np.array([1, 0, 0, 0])
            elif np.linalg.norm(a+b) < 1e-6:
                pose = np.array([0, 0, 0, 1])
            else:
                v = np.cross(a, b) #rotation along v
                theta = np.arccos(np.dot(a, b))
                pose = axangle2quat(v, theta)
            assert np.linalg.norm(rotate_vector(np.array([1, 0, 0]), pose) - b) < 1e-5
            return l, pose

        if isinstance(vec, str):
            vec = np.array(list(map(float, vec.split(' '))))
        l, pose = vec2pose((vec[3:]-vec[:3])/2)
        self.add_capsule(link, (vec[3:] + vec[:3])/2, pose, size, l, rgb, name)

        if density != 1000:
            #self.builder.update_link_mass_and_inertia(link, density)
            raise NotImplementedError

    def add_force_actuator(self, name, low, high):
        # pass
        self.force_actuators.append([name, low, high])

    """
    def state_vector(self):
        raise NotImplementedError
        return np.concatenate([
            self.body_link.get_global_pose().p,
            self.body_link.get_global_pose().q,
            self.wrapper.get_qpos().flat,
            self.body_link.get_linear_velocity(),
            self.body_link.get_angular_velocity(),
            self.wrapper.get_qvel().flat,
            np.clip(self.wrapper.get_cfrc_ext(), -1, 1).flat
        ])
        """

    def state_vector(self):
        return np.concatenate([
            self.get_qpos().flat,
            self.get_qvel().flat,
        ])