from gym import utils, spaces
# from gym.envs.mujoco import mujoco_env
from .sapien_env import SapienEnv
import numpy as np

class SwimmerEnv(SapienEnv, utils.EzPickle):
    def __init__(self):
        SapienEnv.__init__(self, 4, 0.01)
        utils.EzPickle.__init__(self)

    def build_render(self):
        self.sim.set_ambient_light([.4, .4, .4])
        self.sim.set_shadow_light([1, -1, -1], [.5, .5, .5])
        self.sim.add_point_light([2, 2, 2], [1, 1, 1])
        self.sim.add_point_light([2, -2, 2], [1, 1, 1])
        self.sim.add_point_light([-2, 0, 2], [1, 1, 1])

        self._renderer.camera.set_position(np.array([0, -5, 5]))
        self._renderer.camera.rotate_yaw_pitch(0, -0.5)
        self._renderer.set_current_scene(self.sim)
        return self._renderer

    def build_model(self):
        builder = self.builder
        PxIdentity = np.array([1, 0, 0, 0])

        x2y = np.array([0.7071068, 0, 0, 0.7071068])
        x2z = np.array([0.7071068, 0, 0.7071068, 0])


        # TODO: default friction... should be closed
        root1 = self.my_add_link(None,  (np.array([0, 0, 0]), PxIdentity), None, "root1")
        root2 = self.my_add_link(root1, ([0, 0, 0], PxIdentity), ([0, 0, 0], PxIdentity), "root2", "slider1",
                                 [-np.inf, np.inf], type='slider', father_pose_type='sapien')

        root3 = self.my_add_link(root2, ([0, 0, 0], x2y), ([0, 0, 0], x2y), "root3", "slider2",
                                 [-np.inf, np.inf], type='slider', father_pose_type='sapien')

        torso = self.my_add_link(root3, ([0, 0, 0.], x2z), ([0., 0, 0], x2z), "torso", "rot",
                                 [-np.inf, np.inf], type='hinge', father_pose_type='sapien')

        density = 1000.
        self.fromto(torso, "1.5 0 0 0.5 0 0", 0.1, np.array([1., 0, 0]), "torso", density=density)

        range = [np.radians(-100), np.radians(100)]
        mid = self.my_add_link(torso, ([0.5, 0, 0], [1, 0, 0, 0]), ([0, 0, 0], x2z), "mid", "rot2", range)
        self.fromto(mid, "0 0 0 -1 0 0", 0.1, np.array([0., 1., 0]), "mid", density=density)

        back = self.my_add_link(mid, ([-1., 0, 0], [1, 0, 0, 0]), ([0, 0, 0], x2z), "back", "rot3", range)
        self.fromto(back, "0 0 0 -1 0 0", 0.1, np.array([0., 0., 1.]), "back", density=density)

        wrapper = builder.build(True) #fix base = True
        self.add_force_actuator("rot2", -100, 100)
        self.add_force_actuator("rot3", -100, 100)

        self.sim.add_ground(-1)
        return wrapper, None


    def step(self, a):
        ctrl_cost_coeff = 0.0001
        xposbefore = self.model.get_qpos()[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.model.get_qpos()[0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.get_qpos()[2:].flat,
            self.get_qvel().flat,
        ])


    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=len(self.init_qpos)),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=len(self.init_qvel))
        )
        return self._get_obs()
