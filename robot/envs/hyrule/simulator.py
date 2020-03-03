import numpy as np
from gym.utils import seeding
from transforms3d.quaternions import qmult, rotate_vector, axangle2quat
from robot.envs.sapien.camera import CameraRender
from collections import OrderedDict
from gym import Env
from gym.spaces import Box

DEFAULT_SIZE = 500

import sapien.core as sapien_core
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


def load_sapien_state(object):
    if isinstance(object, sapien_core.pysapien.Articulation):
        return {
            'qpos': object.get_qpos(),
            'qvel': object.get_qvel(),
            'qf': object.get_qf(),
            'pose': object.get_links()[0].pose,
        }
    elif isinstance(object, sapien_core.pysapien.Actor):
        return {
            'pose': object.pose,
            'velocity': object.velocity,
            'angular_velocity': object.angular_velocity,
        }
    else:
        raise NotImplementedError

def set_sapien_state(object, state):
    if isinstance(object, sapien_core.pysapien.Articulation):
        object.set_qpos(state['qpos'])
        object.set_qvel(state['qvel'])
        object.set_qf(state['qf'])
        object.set_root_pose(state['pose'])
    elif isinstance(object, sapien_core.pysapien.Actor):
        object.set_pose(state['pose'])
        object.set_velocity(state['velocity'])
        object.set_angular_velocity(state['angular_velocity'])
    else:
        raise NotImplementedError


class Simulator(Env):
    """
    Major interface...
    """
    def __init__(self, dt=0.01, frameskip=1, gravity=(0, 0, -9.8), sim=None):
        self.dt = dt
        self.frameskip = frameskip
        self.viewer = None
        self._viewers = OrderedDict()

        self.metadata = OrderedDict({
            'render.modes': ['human'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        })

        # --------------------- constrain system .................
        if sim is None:
            self.sim = sapien_core.Simulation()
            self._optifuser = sapien_core.OptifuserRenderer()
            self.sim.set_renderer(self._optifuser)
        else:
            self.sim = sim
            self._optifuser = self.sim.get_renderer()
        self.scene: sapien_core.Scene = self.sim.create_scene(gravity=np.array(gravity), solver_type=sapien_core.SolverType.PGS)
        self.scene.set_timestep(dt)

        self.seed()

        self.agent = None # agent is always special in the scene, it should be the only articulation object
        self.objects = OrderedDict()
        self.kinematic_objects = OrderedDict()

        # actuator system ........................................
        self._actuator_range = OrderedDict()
        self._actuator_dof = OrderedDict()
        self._actuator_joint = OrderedDict()
        self._ee_link_idx = OrderedDict()

        self._lock_dof = OrderedDict()
        self._lock_value = OrderedDict()

        self.timestep = 0
        self.costs = None
        self._reset = False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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

    def render(self, mode='human', sleep=0):
        tmp = self._get_viewer(mode).render()
        if sleep > 0:
            import time
            time.sleep(sleep)
        return tmp

    def state_dict(self):
        return dict([(name, load_sapien_state(obj)) for name, obj in self.objects.items()])

    def load_state_dict(self, dict):
        for name, value in dict.items():
            set_sapien_state(self.objects[name], value)

    def state_vector(self):
        return np.concatenate([[self.timestep]] + [np.array(obj.pack()) for name, obj in self.objects.items()])

    def load_state_vector(self, vec):
        self.timestep = int(vec[0])

        vec = vec[1:]
        l = 0
        for name, obj in self.objects.items():
            r = l + len(obj.pack())
            obj.unpack(vec[l:r])
            l = r

    def do_simulation(self):
        self.scene.step()
        self.timestep += 1

        for name, item in self._lock_dof.items():
            q = self.objects[name].get_qpos()
            q[item] = self._lock_value[name]
            self.objects[name].set_qpos(q)

    def reset(self):
        # pass
        if not self._reset:
            self._start_state = self.state_vector().copy()
            self.observation_space = Box(-np.inf, np.inf, self._start_state.shape)
            self.action_space = Box(-1, 1, self._actuator_joint['agent'].shape)

        self.load_state_vector(self._start_state)
        self._reset = True
        self.timestep = 0

    def step(self, action):
        # do_simulation
        assert self._reset

        if len(self._actuator_dof['agent']) > 0:
            action = np.array(action).clip(-1, 1)
            qf = np.zeros(self.agent.dof)
            #TODO: we should not multiply them together
            qf[self._actuator_dof['agent']] = action * self._actuator_range['agent'][:, 0]
            self.agent.set_qf(qf)

        for i in range(self.frameskip):
            self.do_simulation()
        reward = 0
        if self.costs is not None:
            reward = - self.costs.cost(self)
        return self._get_obs(), reward, False, {}

    def _get_obs(self):
        return self.state_vector()

    def __del__(self):
        self.scene = None

    def viewer_setup(self):
        self.scene.set_ambient_light([.4, .4, .4])
        self.scene.set_shadow_light([1, -1, -1], [.5, .5, .5])
        self.scene.add_point_light([2, 2, 2], [1, 1, 1])
        self.scene.add_point_light([2, -2, 2], [1, 1, 1])
        self.scene.add_point_light([-2, 0, 2], [1, 1, 1])

        self._renderer.set_camera_position(3, -1.5, 1.65)
        self._renderer.set_camera_rotation(-3.14 - 0.5, -0.2)
        self._renderer.set_current_scene(self.scene)
