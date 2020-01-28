import os
import copy
import numpy as np
import warnings

import gym
from gym import error, spaces
from gym.utils import seeding
from .path_utils import get_assets_path

try:
    import sapien.core as sapien_core
    from sapien.core import Pose
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install sapien.)".format(e))

DEFAULT_SIZE = 500

from ..control.sapien_env import SapienEnv

class RobotEnv(gym.GoalEnv, SapienEnv):
    def __init__(self, model_path, initial_qpos, n_actions, n_substeps=1):
        if model_path.startswith('/'):
            fullpath = model_path
        else:
            fullpath = os.path.join(get_assets_path(), "robot", f"{model_path}.urdf")
        if not os.path.exists(fullpath):
            raise IOError('File {} does not exist'.format(fullpath))

        # sim/sapien
        self._sim = sapien_core.Simulation()
        self._renderer2 = sapien_core.OptifuserRenderer()
        self._sim.set_renderer(self._renderer2)

        # scene/sim
        self.sim: sapien_core.Scene = self._sim.create_scene()
        self.sim.add_ground(0)
        self.sim.set_timestep(1 / 240)
        self.n_substeps = n_substeps # frameskip


        # load robot
        self._actuator_index = np.array([], dtype=np.int)
        self._actuator_range = np.zeros([0, 2])
        self._q_names = None

        self.loader: sapien_core.URDFLoader = self.sim.create_urdf_loader()
        self.builder = self.sim.create_articulation_builder()
        movo_material = self._sim.create_physical_material(3.0, 2.0, 0.01)
        self.model = self._load_robot(fullpath, movo_material) # get self.model
        self.scene = self._load_scene()


        # setup viewer
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }


        # do the first...
        self.seed()
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.get_state())
        #self.initial_state = np.stack(
        #    [self.model.get_qpos(), self.model.get_qvel(), self.model.get_qacc(), self.model.get_qf()], axis=0)
        #self.initial_state = self.sim.getS

        self.goal = self._sample_goal()
        obs = self._get_obs()

        #assert n_actions == len(self._actuator_range)
        #self.action_space = spaces.Box(-1., 1., shape=(n_actions,), dtype='float32')
        self.action_space = spaces.Box(low=self._actuator_range[:, 0], high=self._actuator_range[:, 1],
                                       dtype=np.float32)
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

    def _load_robot(self, urdf_path: str, material: sapien_core.PxMaterial):
        raise NotImplementedError

    def _load_scene(self) -> None:
        raise NotImplementedError

    def add_force_actuator(self, joint_name: str, low: float, high: float):
        # TODO: support ball joint
        if joint_name not in self._q_names:
            warnings.warn("Joint name {} is not a valid joint for robot.".format(joint_name))
            return
        joint_index = self._q_names.index(joint_name)
        if joint_index in self._actuator_index:
            warnings.warn("Joint with Index {} have already been added an actuator".format(joint_index))
            return
        self._actuator_index = np.append(self._actuator_index, [joint_index])
        self._actuator_range = np.append(self._actuator_range, np.array([[low, high]]), axis=0)


    @property
    def dt(self):
        #return self.sim.model.opt.timestep * self.sim.nsubsteps
        return self.sim.timestep * self.n_substeps

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def get_state(self):
        return np.stack(
            [self.model.get_qpos(), self.model.get_qvel(), self.model.get_qacc(), self.model.get_qf()], axis=0)

    def set_state(self, state):
        #TODO: global pose/object pose
        for i in range(2):
            self.model.set_qpos(state[0, :])
            self.model.set_qvel(state[1, :])
            self.model.set_qacc(state[2, :])
            self.model.set_qf(state[3, :])

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        super(RobotEnv, self).reset()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        return obs

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        self._render_callback()
        return self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                #self.viewer = mujoco_py.MjViewer(self.sim)
                self.viewer = sapien_core.OptifuserController(self._renderer2)
            elif mode == 'rgb_array':
                from ..camera import CameraRender
                self.viewer = CameraRender(self.sim, mode, width=512, height=512)

            self._viewer_setup()
            if mode == 'human':
                self.viewer.show_window()

            self.viewer.set_current_scene(self.sim)
            self._viewers[mode] = self.viewer

        self.sim.update_render()
        return self.viewer

    # Extension methods
    # ----------------------------

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        self.set_state(self.initial_state)
        #self.sim.forward()
        # forward -> step?
        self.sim.step()
        return True

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass
