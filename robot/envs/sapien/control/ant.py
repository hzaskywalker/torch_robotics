from gym import utils, spaces
#from gym.envs.mujoco import mujoco_env
from .sapien_env import Pose, sapien_core, SapienEnv
import numpy as np

class AntEnv(SapienEnv, utils.EzPickle):
    def __init__(self):
        SapienEnv.__init__(self, 5, 0.01)
        utils.EzPickle.__init__(self)

    def build_render(self):
        renderer = sapien_core.OptifuserController(self._renderer2)

        self.sim.set_ambient_light([.4, .4, .4])
        self.sim.set_shadow_light([1, -1, -1], [.5, .5, .5])
        self.sim.add_point_light([2, 2, 2], [1, 1, 1])
        self.sim.add_point_light([2, -2, 2], [1, 1, 1])
        self.sim.add_point_light([-2, 0, 2], [1, 1, 1])

        renderer.camera.set_position(np.array([0, 1, 10]))
        renderer.camera.set_forward(np.array([0, 1, 0]))
        renderer.camera.set_up(np.array([0, 0, 1]))
        renderer.camera.rotate_yaw_pitch(0, -1.5)
        return renderer

    def build_model(self):
        builder = self.builder
        PxIdentity = np.array([1, 0, 0, 0])
        density = 5
        body_link = builder.add_link(None, Pose(np.array([0, 0, 0]), PxIdentity), "body")
        builder.add_sphere_shape_to_link(body_link, Pose(np.array([0, 0, 0]), PxIdentity), 0.25)
        builder.add_sphere_visual_to_link(body_link, Pose(np.array([0, 0, 0]), PxIdentity), 0.25)
        builder.add_capsule_shape_to_link(body_link,  Pose(np.array([0.141, 0, 0]), PxIdentity), 0.08, 0.141)
        builder.add_capsule_visual_to_link(body_link, Pose(np.array([0.141, 0, 0]), PxIdentity), 0.08, 0.141)
        builder.add_capsule_shape_to_link(body_link,  Pose(np.array([-0.141, 0, 0]), PxIdentity), 0.08, 0.141)
        builder.add_capsule_visual_to_link(body_link, Pose(np.array([-0.141, 0, 0]), PxIdentity), 0.08, 0.141)
        builder.add_capsule_shape_to_link(body_link, Pose(np.array([0, 0.141, 0]), np.array([0.707107, 0, 0, 0.707107])), 0.08, 0.141)
        builder.add_capsule_visual_to_link(body_link, Pose(np.array([0, 0.141, 0]), np.array([0.707107, 0, 0, 0.707107])), 0.08, 0.141)
        builder.add_capsule_shape_to_link(body_link, Pose(np.array([0, -0.141, 0]), np.array([0.707107, 0, 0, 0.707107])), 0.08, 0.141)
        builder.add_capsule_visual_to_link(body_link, Pose(np.array([0, -0.141, 0]), np.array([0.707107, 0, 0, 0.707107])), 0.08, 0.141)
        builder.update_link_mass_and_inertia(body_link, density)
        l1 = builder.add_link(body_link,  Pose(np.array([0, 0, 0]), PxIdentity), "l1", "j1",
                                    sapien_core.PxArticulationJointType.REVOLUTE, np.array([[-0.5236, 0.5236]]),
                                    Pose(np.array([0.282, 0, 0]), np.array([0.7071068, 0, 0.7071068, 0])),
                                    Pose(np.array([0.141, 0, 0]), np.array([0.7071068, 0, -0.7071068, 0])))
        builder.add_capsule_shape_to_link(l1,  Pose(np.array([0, 0, 0]), PxIdentity), 0.08, 0.141)
        builder.add_capsule_visual_to_link(l1,  Pose(np.array([0, 0, 0]), PxIdentity), 0.08, 0.141)
        builder.update_link_mass_and_inertia(l1, density)
        l2 = builder.add_link(body_link,  Pose(np.array([0, 0, 0]), PxIdentity), "l2", "j2",
                                    sapien_core.PxArticulationJointType.REVOLUTE, np.array([[-0.5236, 0.5236]]),
                                    Pose(np.array([-0.282, 0, 0]), np.array([0, 0.7071068, 0, -0.7071068])),
                                    Pose(np.array([0.141, 0, 0]), np.array([0.7071068, 0, -0.7071068, 0])))
        builder.add_capsule_shape_to_link(l2,  Pose(np.array([0, 0, 0]), PxIdentity), 0.08, 0.141)
        builder.add_capsule_visual_to_link(l2,  Pose(np.array([0, 0, 0]), PxIdentity), 0.08, 0.141)
        builder.update_link_mass_and_inertia(l2, density)
        l3 = builder.add_link(body_link,  Pose(np.array([0, 0, 0]), PxIdentity), "l3", "j3",
                                    sapien_core.PxArticulationJointType.REVOLUTE, np.array([[-0.5236, 0.5236]]),
                                    Pose(np.array([0, 0.282, 0]), np.array([0.5, -0.5, 0.5, 0.5])),
                                    Pose(np.array([0.141, 0, 0]), np.array([0.7071068, 0, -0.7071068, 0])))
        builder.add_capsule_shape_to_link(l3,  Pose(np.array([0, 0, 0]), PxIdentity), 0.08, 0.141)
        builder.add_capsule_visual_to_link(l3,  Pose(np.array([0, 0, 0]), PxIdentity), 0.08, 0.141)
        builder.update_link_mass_and_inertia(l3, density)
        l4 = builder.add_link(body_link,  Pose(np.array([0, 0, 0]), PxIdentity), "l4", "j4",
                                    sapien_core.PxArticulationJointType.REVOLUTE, np.array([[-0.5236, 0.5236]]),
                                    Pose(np.array([0, -0.282, 0]), np.array([0.5, 0.5, 0.5, -0.5])),
                                    Pose(np.array([0.141, 0, 0]), np.array([ 0.7071068, 0, -0.7071068, 0])))
        builder.add_capsule_shape_to_link(l4,  Pose(np.array([0, 0, 0]), PxIdentity), 0.08, 0.141)
        builder.add_capsule_visual_to_link(l4,  Pose(np.array([0, 0, 0]), PxIdentity), 0.08, 0.141)
        builder.update_link_mass_and_inertia(l4, density)
        f1 = builder.add_link(l1,  Pose(np.array([0, 0, 0]), PxIdentity), "f1", "j11",
                                    sapien_core.PxArticulationJointType.REVOLUTE, np.array([[0.5236, 1.222]]),
                                    Pose(np.array([-0.141, 0, 0]), np.array([0, 0.7071068, 0.7071068, 0])),
                                    Pose(np.array([0.282, 0, 0]), np.array([0, 0.7071068, 0.7071068, 0])))
        builder.add_capsule_shape_to_link(f1,  Pose(np.array([0, 0, 0]), PxIdentity), 0.08, 0.282)
        builder.add_capsule_visual_to_link(f1,  Pose(np.array([0, 0, 0]), PxIdentity), 0.08, 0.282)
        builder.update_link_mass_and_inertia(f1, density)
        f2 = builder.add_link(l2,  Pose(np.array([0, 0, 0]), PxIdentity), "f2", "j21",
                                    sapien_core.PxArticulationJointType.REVOLUTE, np.array([[0.5236, 1.222]]),
                                    Pose(np.array([-0.141, 0, 0]), np.array([0, 0.7071068, 0.7071068, 0])),
                                    Pose(np.array([0.282, 0, 0]), np.array([0, 0.7071068, 0.7071068, 0])))
        builder.add_capsule_shape_to_link(f2,  Pose(np.array([0, 0, 0]), PxIdentity), 0.08, 0.282)
        builder.add_capsule_visual_to_link(f2,  Pose(np.array([0, 0, 0]), PxIdentity), 0.08, 0.282)
        builder.update_link_mass_and_inertia(f2, density)
        f3 = builder.add_link(l3,  Pose(np.array([0, 0, 0]), PxIdentity), "f3", "j31",
                                    sapien_core.PxArticulationJointType.REVOLUTE, np.array([[0.5236, 1.222]]),
                                    Pose(np.array([-0.141, 0, 0]), np.array([0, 0.7071068, 0.7071068, 0])),
                                    Pose(np.array([0.282, 0, 0]), np.array([0, 0.7071068, 0.7071068, 0])))
        builder.add_capsule_shape_to_link(f3,  Pose(np.array([0, 0, 0]), PxIdentity), 0.08, 0.282)
        builder.add_capsule_visual_to_link(f3,  Pose(np.array([0, 0, 0]), PxIdentity), 0.08, 0.282)
        builder.update_link_mass_and_inertia(f3, density)
        f4 = builder.add_link(l4,  Pose(np.array([0, 0, 0]), PxIdentity), "f4", "j41",
                                    sapien_core.PxArticulationJointType.REVOLUTE, np.array([[0.5236, 1.222]]),
                                    Pose(np.array([-0.141, 0, 0]), np.array([0, 0.7071068, 0.7071068, 0])),
                                    Pose(np.array([0.282, 0, 0]), np.array([0, 0.7071068, 0.7071068, 0])))
        builder.add_capsule_shape_to_link(f4,  Pose(np.array([0, 0, 0]), PxIdentity), 0.08, 0.282)
        builder.add_capsule_visual_to_link(f4,  Pose(np.array([0, 0, 0]), PxIdentity), 0.08, 0.282)
        builder.update_link_mass_and_inertia(f4, density)
        wrapper = builder.build(False)
        lower_bound = -10
        upper_bound = 10
        wrapper.add_force_actuator("j1", lower_bound, upper_bound)
        wrapper.add_force_actuator("j2", lower_bound, upper_bound)
        wrapper.add_force_actuator("j3", lower_bound, upper_bound)
        wrapper.add_force_actuator("j4", lower_bound, upper_bound)
        wrapper.add_force_actuator("j11", lower_bound, upper_bound)
        wrapper.add_force_actuator("j21", lower_bound, upper_bound)
        wrapper.add_force_actuator("j31", lower_bound, upper_bound)
        wrapper.add_force_actuator("j41", lower_bound, upper_bound)
        wrapper.set_root_pose(np.array([0, 0, 0.55]))
        ground = self.sim.add_ground(-1)

        self.body_link = body_link
        self.init_root_pose_p = self.body_link.get_global_pose().p
        self.init_root_pose_q = self.body_link.get_global_pose().q

        return wrapper, body_link

    def step(self, a):
        xposbefore = self.body_link.get_global_pose().p[0] #torso
        self.do_simulation(a, self.frame_skip)
        xposafter = self.body_link.get_global_pose().p[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.model.get_cfrc_ext(), -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = state[2:]#self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):

        return np.concatenate([
            self.body_link.get_global_pose().p[2:],
            self.body_link.get_global_pose().q,
            self.model.get_qpos().flat[2:],
            self.body_link.get_linear_velocity(),
            self.body_link.get_angular_velocity(),
            self.model.get_qvel().flat,
            np.clip(self.model.get_cfrc_ext(), -1, 1).flat, # 54->84 ??
        ])

    def reset_model(self):
        qpos = self.init_qpos
        qpos[7:] += self.np_random.uniform(size=8, low=-.1, high=.1)
        qvel = self.init_qvel
        qvel[6:] += self.np_random.uniform(size=8, low=-.1, high=.1)
        #self.model.set_qpos(qpos)
        #self.model.set_qvel(qvel)
        #root_pose_p = self.init_root_pose_p #+ self.np_random.uniform(size=3, low=-.1, high=.1)
        #root_pose_q = self.init_root_pose_q #+ self.np_random.uniform(size=4, low=-.1, high=.1)
        #self.model.set_root_pose(root_pose_p, root_pose_q)
        # TODO: reset root_velocity
        self.set_state(qpos, qvel)
        return self._get_obs()

    #def viewer_setup(self):
        #self.viewer.cam.distance = self.model.stat.extent * 0.5
        #pass
