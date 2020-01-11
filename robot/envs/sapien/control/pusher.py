import numpy as np
from .sapien_env import Pose, sapien_core, SapienEnv
from gym import utils

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class PusherEnv(SapienEnv, utils.EzPickle):
    def __init__(self):
        SapienEnv.__init__(self, 5)
        utils.EzPickle.__init__(self)

    def build_render(self):
        renderer = sapien_core.OptifuserRenderer()

        renderer.set_ambient_light([.4, .4, .4])
        renderer.set_shadow_light([1, -1, -1], [.5, .5, .5])
        renderer.add_point_light([2, 2, 2], [1, 1, 1])
        renderer.add_point_light([2, -2, 2], [1, 1, 1])
        renderer.add_point_light([-2, 0, 2], [1, 1, 1])

        renderer.cam.set_position(np.array([4, 0, 4]))
        renderer.cam.rotate_yaw_pitch(np.pi/2, -0.5)
        return renderer

    def build_model(self):
        builder = self.builder
        PxIdentity = np.array([1, 0, 0, 0])
        x2y = np.array([0.7071068, 0, 0, 0.7071068])
        x2z = np.array([0.7071068, 0, 0.7071068, 0])
        rgb = np.array([0.1, 0.1, 0.1])
        rgb2 = np.array([0.6, 0.6, 0.6])
        default_rgb = np.array([0.5, 0.5, 0.5])

        cur = world = builder.add_link(None,  Pose(np.array([0, 0, 0]), PxIdentity), "world") # root coordinates #free
        cur = r_shoulder_pan_link = self.my_add_link(cur, ([0, -0.6, 0], PxIdentity), ([0, 0, 0], x2z),
                                               "r_shoulder_pan_link", "r_shoulder_pan_joint", [-2.2854, 1.714602])
        self.add_sphere(builder, cur, np.array([-0.06, 0.05, 0.2]), PxIdentity, 0.05, rgb2, "e1")
        self.add_sphere(builder, cur,  np.array([0.06, 0.05, 0.2]), PxIdentity, 0.05, rgb2, "e2")
        self.add_sphere(builder, cur, np.array([-0.06, 0.09, 0.2]), PxIdentity, 0.03, rgb, "e1p")
        self.add_sphere(builder, cur, np.array([0.06, 0.09, 0.2]), PxIdentity, 0.03, rgb, "e2p")
        self.fromto(cur, "0 0 -0.4 0 0 0.2", 0.1, default_rgb, "sp")


        cur = r_shoulder_lift_link = self.my_add_link(cur, ([0.1, 0, 0], PxIdentity), ([0, 0, 0], x2y),
                                               "r_shoulder_lift_link", "r_shoulder_lift_joint", [-0.5236, 1.3963])
        self.fromto(cur, "0 -0.1 0 0 0.1 0", 0.1, default_rgb, "sl")

        cur = r_upper_arm_roll_link = self.my_add_link(cur, ([0, 0, 0], PxIdentity), ([0, 0, 0], PxIdentity),
                                               "r_upper_arm_roll_link", "r_upper_arm_roll_joint", [-1.5, 1.7])

        self.fromto(cur, "-0.1 0 0 0.1 0 0", 0.02, default_rgb, "uar")
        self.fromto(cur, "0 0 0 0.4 0 0", 0.06, default_rgb, "ua")

        cur = r_elbow_flex_link = self.my_add_link(cur, ([0.4, 0, 0], PxIdentity), ([0, 0, 0], x2y),
                                                 "r_elbow_flex_link", "r_elbow_flex_joint", [-2.3213, 0])
        self.fromto(cur, "0 -0.02 0 0.0 0.02 0", 0.06, default_rgb, "ef")

        cur = r_forearm_roll_link = self.my_add_link(cur, ([0, 0, 0], PxIdentity), ([0, 0, 0], PxIdentity),
                                             "r_forearm_roll_link", "r_forearm_roll_joint", [-1.5, 1.5])
        self.fromto(cur, "-0.1 0 0 0.1 0 0", 0.02, default_rgb, "fr")
        self.fromto(cur, "0 0 0 0.291 0 0", 0.05, default_rgb, "fa")


        cur = r_wrist_flex_link = self.my_add_link(cur, ([0.321, 0, 0], PxIdentity), ([0, 0, 0], x2y),
                                               "r_wrist_flex_link", "r_wrist_flex_joint", [-1.094, 0.])
        self.fromto(cur, "0 -0.02 0 0 0.02 0", 0.01, rgb, "wf")

        cur = r_wrist_roll_link = self.my_add_link(cur, ([0., 0, 0], PxIdentity), ([0, 0, 0], PxIdentity),
                                             "r_wrist_roll_link", "r_wrist_roll_joint", [-1.5, 1.5])

        self.add_sphere(builder, cur, np.array([0.1, -0.1, 0.]), PxIdentity, 0.01, default_rgb, "tip_arml")
        self.add_sphere(builder, cur, np.array([0.1, 0.1, 0.]), PxIdentity, 0.01, default_rgb, "tip_armr")

        self.fromto(cur, "0 -0.1 0. 0.0 +0.1 0", 0.02, default_rgb, "hand1")
        self.fromto(cur, "0 -0.1 0. 0.1 -0.1 0", 0.02, default_rgb, "hand2")
        self.fromto(cur, "0 +0.1 0. 0.1 +0.1 0", 0.02, default_rgb, "hand3")


        wrapper = builder.build(True) #fix base = True
        #wrapper.add_force_actuator("abdomen_z", -100, 100)
        ground = self.sim.add_ground(-1)
        return wrapper, None


    def step(self, a):
        #vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        #vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        #reward_near = - np.linalg.norm(vec_1)
        #reward_dist = - np.linalg.norm(vec_2)
        #reward_ctrl = - np.square(a).sum()
        reward_near = 0
        reward_dist = 0
        reward_ctrl = 0
        reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist,
                reward_ctrl=reward_ctrl)


    def reset_model(self):
        return self._get_obs()

        qpos = self.init_qpos

        self.goal_pos = np.asarray([0, 0])
        while True:
            self.cylinder_pos = np.concatenate([
                    self.np_random.uniform(low=-0.3, high=0, size=1),
                    self.np_random.uniform(low=-0.2, high=0.2, size=1)])
            if np.linalg.norm(self.cylinder_pos - self.goal_pos) > 0.17:
                break

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                high=0.005, size=len(self.init_qvel))
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.model.get_qpos().flat[:7],
            self.model.get_qpos().flat[:7],
            #self.get_body_com("tips_arm"),
            #self.get_body_com("object"),
            #self.get_body_com("goal"),
        ])