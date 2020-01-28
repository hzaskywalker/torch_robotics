from gym import utils, spaces
#from gym.envs.mujoco import mujoco_env
from .sapien_env import Pose, sapien_core, SapienEnv
import numpy as np
from .sapien_env import sapien_core as pysapien

from transforms3d.quaternions import axangle2quat as aa

class AntEnv(SapienEnv, utils.EzPickle):
    def __init__(self):
        SapienEnv.__init__(self, 5, 0.01)
        utils.EzPickle.__init__(self)

    def build_render(self):
        self.sim.set_ambient_light([.4, .4, .4])
        self.sim.set_shadow_light([1, -1, -1], [.5, .5, .5])
        self.sim.add_point_light([2, 2, 2], [1, 1, 1])
        self.sim.add_point_light([2, -2, 2], [1, 1, 1])
        self.sim.add_point_light([-2, 0, 2], [1, 1, 1])

        self._renderer.set_camera_position(0, 1, 10)
        #self._renderer.camera.set_forward(np.array([0, 1, 0]))
        #self._renderer.camera.set_up(np.array([0, 0, 1]))
        self._renderer.set_camera_rotation(0, -1.5)

    def build_model(self):
        builder = self.builder
        body = builder.create_link_builder()
        body.add_sphere_shape(Pose(), 0.25)
        body.add_sphere_visual(Pose(), 0.25)
        body.add_capsule_shape(Pose([0.141, 0, 0]), 0.08, 0.141)
        body.add_capsule_visual(Pose([0.141, 0, 0]), 0.08, 0.141)
        body.add_capsule_shape(Pose([-0.141, 0, 0]), 0.08, 0.141)
        body.add_capsule_visual(Pose([-0.141, 0, 0]), 0.08, 0.141)
        body.add_capsule_shape(Pose([0, 0.141, 0], aa([0, 0, 1], np.pi / 2)), 0.08, 0.141)
        body.add_capsule_visual(Pose([0, 0.141, 0], aa([0, 0, 1], np.pi / 2)), 0.08, 0.141)
        body.add_capsule_shape(Pose([0, -0.141, 0], aa([0, 0, 1], np.pi / 2)), 0.08, 0.141)
        body.add_capsule_visual(Pose([0, -0.141, 0], aa([0, 0, 1], np.pi / 2)), 0.08, 0.141)
        body.set_name("body")

        l1 = builder.create_link_builder(body)
        l1.set_name("l1")
        l1.set_joint_name("j1")
        l1.set_joint_properties(pysapien.ArticulationJointType.REVOLUTE, [[-0.5236, 0.5236]],
                                Pose([0.282, 0, 0], [0.7071068, 0, 0.7071068, 0]),
                                Pose([0.141, 0, 0], [-0.7071068, 0, 0.7071068, 0]), 0.1)
        l1.add_capsule_shape(Pose(), 0.08, 0.141)
        l1.add_capsule_visual(Pose(), 0.08, 0.141)

        l2 = builder.create_link_builder(body)
        l2.set_name("l2")
        l2.set_joint_name("j2")
        l2.set_joint_properties(pysapien.ArticulationJointType.REVOLUTE, [[-0.5236, 0.5236]],
                                Pose([-0.282, 0, 0], [0, -0.7071068, 0, 0.7071068]),
                                Pose([0.141, 0, 0], [-0.7071068, 0, 0.7071068, 0]), 0.1)
        l2.add_capsule_shape(Pose(), 0.08, 0.141)
        l2.add_capsule_visual(Pose(), 0.08, 0.141)

        l3 = builder.create_link_builder(body)
        l3.set_name("l3")
        l3.set_joint_name("j3")
        l3.set_joint_properties(pysapien.ArticulationJointType.REVOLUTE, [[-0.5236, 0.5236]],
                                Pose([0, 0.282, 0], [0.5, -0.5, 0.5, 0.5]),
                                Pose([0.141, 0, 0], [0.7071068, 0, -0.7071068, 0]), 0.1)
        l3.add_capsule_shape(Pose(), 0.08, 0.141)
        l3.add_capsule_visual(Pose(), 0.08, 0.141)

        l4 = builder.create_link_builder(body)
        l4.set_name("l4")
        l4.set_joint_name("j4")
        l4.set_joint_properties(pysapien.ArticulationJointType.REVOLUTE, [[-0.5236, 0.5236]],
                                Pose([0, -0.282, 0], [0.5, 0.5, 0.5, -0.5]),
                                Pose([0.141, 0, 0], [0.7071068, 0, -0.7071068, 0]), 0.1)
        l4.add_capsule_shape(Pose(), 0.08, 0.141)
        l4.add_capsule_visual(Pose(), 0.08, 0.141)

        f1 = builder.create_link_builder(l1)
        f1.set_name("f1")
        f1.set_joint_name("j11")
        f1.set_joint_properties(pysapien.ArticulationJointType.REVOLUTE, [[0.5236, 1.222]],
                                Pose([-0.141, 0, 0], [0, 0.7071068, 0.7071068, 0]),
                                Pose([0.282, 0, 0], [0, 0.7071068, 0.7071068, 0]), 0.1)
        f1.add_capsule_shape(Pose(), 0.08, 0.282)
        f1.add_capsule_visual(Pose(), 0.08, 0.282)

        f2 = builder.create_link_builder(l2)
        f2.set_name("f2")
        f2.set_joint_name("j21")
        f2.set_joint_properties(pysapien.ArticulationJointType.REVOLUTE, [[0.5236, 1.222]],
                                Pose([-0.141, 0, 0], [0, 0.7071068, 0.7071068, 0]),
                                Pose([0.282, 0, 0], [0, 0.7071068, 0.7071068, 0]), 0.1)
        f2.add_capsule_shape(Pose(), 0.08, 0.282)
        f2.add_capsule_visual(Pose(), 0.08, 0.282)

        f3 = builder.create_link_builder(l3)
        f3.set_name("f3")
        f3.set_joint_name("j31")
        f3.set_joint_properties(pysapien.ArticulationJointType.REVOLUTE, [[0.5236, 1.222]],
                                Pose([-0.141, 0, 0], [0, 0.7071068, 0.7071068, 0]),
                                Pose([0.282, 0, 0], [0, 0.7071068, 0.7071068, 0]), 0.1)
        f3.add_capsule_shape(Pose(), 0.08, 0.282)
        f3.add_capsule_visual(Pose(), 0.08, 0.282)

        f4 = builder.create_link_builder(l4)
        f4.set_name("f4")
        f4.set_joint_name("j41")
        f4.set_joint_properties(pysapien.ArticulationJointType.REVOLUTE, [[0.5236, 1.222]],
                                Pose([-0.141, 0, 0], [0, 0.7071068, 0.7071068, 0]),
                                Pose([0.282, 0, 0], [0, 0.7071068, 0.7071068, 0]), 0.1)
        f4.add_capsule_shape(Pose(), 0.08, 0.282)
        f4.add_capsule_visual(Pose(), 0.08, 0.282)

        wrapper = builder.build(False)

        lower_bound = -2000
        upper_bound = 2000
        self.add_force_actuator("j1", lower_bound, upper_bound)
        self.add_force_actuator("j2", lower_bound, upper_bound)
        self.add_force_actuator("j3", lower_bound, upper_bound)
        self.add_force_actuator("j4", lower_bound, upper_bound)
        self.add_force_actuator("j11", lower_bound, upper_bound)
        self.add_force_actuator("j21", lower_bound, upper_bound)
        self.add_force_actuator("j31", lower_bound, upper_bound)
        self.add_force_actuator("j41", lower_bound, upper_bound)
        wrapper.set_root_pose(Pose(np.array([0, 0, 0.55])))
        ground = self.sim.add_ground(-1)

        self.body_link = wrapper.get_links()[0]
        self.init_root_pose_p = self.body_link.pose.p
        self.init_root_pose_q = self.body_link.pose.q

        return wrapper, self.body_link

    def step(self, a):
        xposbefore = self.body_link.pose.p[0] #torso
        self.do_simulation(a, self.frame_skip)
        xposafter = self.body_link.pose.p[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        #contact_cost = 0.5 * 1e-3 * np.sum(
        #    np.square(np.clip(self.model.get_cfrc_ext(), -1, 1)))
        #TODO: contact cost, get_cfrc_ext

        contact_cost = 0
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
            self.body_link.pose.p[2:],
            self.body_link.pose.q,
            self.model.get_qpos().flat[2:],
            self.body_link.velocity,
            self.body_link.angular_velocity,
            self.model.get_qvel().flat,
            # TODO: cfrc
            #np.clip(self.model.get_cfrc_ext(), -1, 1).flat, # 54->84 ??
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
