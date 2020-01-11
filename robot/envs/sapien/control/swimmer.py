from gym import utils, spaces
# from gym.envs.mujoco import mujoco_env
from .sapien_env import Pose, sapien_core, SapienEnv
import numpy as np

class SwimmerEnv(SapienEnv, utils.EzPickle):
    def __init__(self):
        SapienEnv.__init__(self, 4, 0.01)
        utils.EzPickle.__init__(self)

    def build_render(self):
        renderer = sapien_core.OptifuserRenderer()

        renderer.set_ambient_light([.4, .4, .4])
        renderer.set_shadow_light([1, -1, -1], [.5, .5, .5])
        renderer.add_point_light([2, 2, 2], [1, 1, 1])
        renderer.add_point_light([2, -2, 2], [1, 1, 1])
        renderer.add_point_light([-2, 0, 2], [1, 1, 1])

        renderer.cam.set_position(np.array([0, -5, 5]))
        renderer.cam.rotate_yaw_pitch(0, -0.5)
        return renderer

    def build_model(self):
        builder = self.builder
        PxIdentity = np.array([1, 0, 0, 0])

        density = 5

        x2y = np.array([0.7071068, 0, 0, 0.7071068])
        x2z = np.array([0.7071068, 0, 0.7071068, 0])


        root1 = builder.add_link(None,  Pose(np.array([0, 0, 0]), PxIdentity), "root1")
        root2 = builder.add_link(root1, Pose(np.array([0, 0, 0]), PxIdentity), "root2", "slider1",
                                 sapien_core.PxArticulationJointType.PRISMATIC, np.array([[-np.inf, np.inf]]),
                                 Pose(np.array([0, 0, 0]), PxIdentity), Pose(np.array([0, 0, 0]), PxIdentity))

        root3 = builder.add_link(root2, Pose(np.array([0, 0, 0]), PxIdentity), "root3", "slider2",
                                 sapien_core.PxArticulationJointType.PRISMATIC, np.array([[-np.inf, np.inf]]),
                                 Pose(np.array([0, 0, 0]), x2y), Pose(np.array([0, 0, 0]), x2y)
                             )
        torso = builder.add_link(root3, Pose(np.array([0, 0, 0]), PxIdentity), "torso", "rot",
                                 sapien_core.PxArticulationJointType.REVOLUTE, np.array([[-np.inf, np.inf]]),
                                 Pose(np.array([0, 0, 0.]), x2z), Pose(np.array([0., 0, 0]), x2z))

        mid = builder.add_link(torso, Pose(np.array([0, 0, 0]), PxIdentity), "mid", "rot2",
                                 sapien_core.PxArticulationJointType.REVOLUTE, np.array([[-100, 100]]),
                                 Pose(np.array([0.5, 0, 0]), x2z), Pose(np.array([0.5, 0, 0]), x2z))

        back = builder.add_link(mid, Pose(np.array([0, 0, 0]), PxIdentity), "back", "rot3",
                               sapien_core.PxArticulationJointType.REVOLUTE, np.array([[-100, 100]]),
                               Pose(np.array([-0.5, 0, 0]), x2z), Pose(np.array([0.5, 0, 0]), x2z))


        self.add_capsule(builder, torso, np.array([1., 0, 0]), PxIdentity, 0.1, 0.5,
                         np.array([1., 0., 0.]), "torso", shape=True)
        self.add_capsule(builder, mid, np.array([0., 0, 0]), PxIdentity, 0.1, 0.5,
                         np.array([0., 1., 0.]), "mid", shape=True)
        self.add_capsule(builder, back, np.array([0., 0, 0, 0]), PxIdentity, 0.1, 0.5,
                         np.array([0., 0., 1.]), "back", shape=True)

        density = 100
        builder.update_link_mass_and_inertia(torso, density)
        builder.update_link_mass_and_inertia(mid, density)
        builder.update_link_mass_and_inertia(back, density)

        # TODO: how to add viscosity
        # TODO: length of capsules is wrong
        # TODO: what's the density and gear in mujoco

        wrapper = builder.build(True) #fix base = True
        wrapper.add_force_actuator("rot2", -30, 30)
        wrapper.add_force_actuator("rot3", -30, 30)

        ground = self.sim.add_ground(-1)
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
