import numpy as np
from gym.utils import seeding
import gym
from gym import spaces
import logging
from inspect import isfunction
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


def load_sapien_state(object):
    if isinstance(object, sapien_core.pysapien.Articulation):
        # TODO: assume that articulation is fully characterized by qpos
        return {
            'qpos': object.get_qpos().flat,
            'qvel': object.get_qvel().flat,
            'qf': object.get_qf().flat
        }
    elif isinstance(object, sapien_core.pysapien.Actor):
        return {
            'pose': object.pose.flat,
            'velocity': object.velocity.flat,
            'angular_velocity': object.get_angular_velocity.flat,
        }
    else:
        raise NotImplementedError

def set_sapien_state(object, state):
    if isinstance(object, sapien_core.pysapien.Articulation):
        object.set_qpos(state['qpos'])
        object.set_qvel(state['qvel'])
        object.set_qf(state['qf'])
    elif isinstance(object, sapien_core.pysapien.Actor):
        object.set_pose(state['pose'])
        object.set_velocity(state['veolicty'])
        object.set_angular_velocity(state['set_angular_velocity'])
    else:
        raise NotImplementedError


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
        self.objects = {}
        self._instr_set = {}
        self.build_scene()

        self._constraints = []


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

    def add_constraints(self, constraint):
        if not constraint.prerequisites(self):
            logging.warning("Add constraint failed")
            return self

        self._constraints.append(constraint)
        k = np.argsort([i.priority for i in self._constraints])
        # sort from priority 0 to 1
        self._constraints = [self._constraints[i] for i in k]
        return self

    def remove_constraints(self, constraints):
        self._constraints.remove(constraints)

    def state_dict(self):
        return dict([(name, load_sapien_state(obj)) for name, obj in self.objects.items()])

    def load_state_dict(self, dict):
        for name, value in dict.items():
            set_sapien_state(self.objects[name], value)
            #self.objects[name].unpack(value)

    def do_simulation(self, constraints=None):
        if constraints is not None:
            for i in constraints:
                i.preprocess(self)
        self.scene.step()
        if constraints is not None:
            for i in constraints:
                i.postprocess(self)
        new_state = self.scene.get_state()
        return new_state

    def num_violation(self, state, constraints):
        self.load_state_dict(state)
        new_state = self.do_constrained_simulation(constraints)
        num_violation = 0
        for i in constraints:
            num_violation += int(not i.satisfy(self, state, new_state))
        return num_violation

    def solve(self, state, constraints):
        # assume constraints are sortest from the small to large
        cur = self.num_violation(state, constraints)
        while cur != 0:
            for i in range(len(constraints)):
                new_constrain = [constraints[i] for j in range(len(constraints)) if j!=i]
                tmp = self.num_violation(state, new_constrain)
                if tmp < cur:
                    cur = tmp
                    constraints = new_constrain
                    break
        return constraints

    def step(self):
        state = self.state_dict()
        constraints = self.solve(state, self._constraints)
        self.do_simulation(constraints)
        self._constraints = [i for i in constraints if i.perpetual]
        return self

    def build_scene(self):
        raise NotImplementedError

    def build_renderer(self):
        raise NotImplementedError

    def __del__(self):
        self.sim = None
        self.scene = None

    def register(self, name, type):
        self._instr_set[name] = type
        return self

    def __getattr__(self, item):
        if item in self.objects:
            return self.objects[item]
        if item in self.instr_set:
            out = self.instr_set[item]
            def run(*args, **kwargs):
                self.add_constraints(out(*args, **kwargs))
                return self
            return run
        else:
            raise AttributeError(f"No registered instruction {item}")

