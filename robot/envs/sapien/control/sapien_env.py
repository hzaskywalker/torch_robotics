from collections import OrderedDict
import numpy as np
from gym.utils import seeding
import gym
from gym import spaces

DEFAULT_SIZE = 500
import sapien

try:
    import sapyen as sapien_core
except ModuleNotFoundError:
    import sapien.core as sapien_core


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

        self.metadata = {
            'render.modes': ['human'], 
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.sim = sapien_core.Simulation()
        self.sim.set_time_step(timestep)

        self._renderer = self.build_render()
        self.sim.set_renderer(self._renderer)

        self.model = self.build_model()

        self.init_qpos = self.model.get_qpos().ravel().copy()
        self.init_qvel = self.model.get_qvel().ravel().copy()

        self._set_action_space()
        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done

        self._set_observation_space(observation)
        self.seed()

        self.create_window = False


    def _set_action_space(self):
        bounds = self.model.get_force_actuator_range().copy()
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
        pass

    # -----------------------------

    def reset(self):
        #self.sim.reset()
        ob = self.reset_model()
        return ob


    def set_state(self, qpos, qvel):
        '''assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()'''
        pass

    @property
    def dt(self):
        return self.timestep * self.frame_skip

    def do_simulation(self, a, n_frames):
        for _ in range(n_frames):
            self.model.apply_actuator(a)
            self.sim.step()


    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        if mode == 'human':
            if not self.create_window:
                self._renderer.show_window()

            self.sim.update_renderer()
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

    def state_vector(self):
        raise NotImplementedError
        """
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