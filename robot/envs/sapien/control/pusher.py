import numpy as np
from .sapien_env import Pose, sapien_core, SapienEnv
from gym import utils

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class PusherEnv(SapienEnv, utils.EzPickle):
    def __init__(self):
        SapienEnv.__init__(self, 10, timestep=0.005, gravity=[0, 0, 0])
        utils.EzPickle.__init__(self)

    def build_render(self):
        self.sim.set_ambient_light([.4, .4, .4])
        self.sim.set_shadow_light([1, -1, -1], [.5, .5, .5])

        self._renderer.set_camera_position(0.8, -2.2, 2)
        self._renderer.set_camera_rotation(-np.pi-np.pi/2, -0.9)

    def build_model(self):
        builder = self.builder
        self._default_density = 300

        PxIdentity = np.array([1, 0, 0, 0])
        x2y = np.array([0.7071068, 0, 0, 0.7071068])
        x2z = np.array([0.7071068, 0, -0.7071068, 0])
        rgb = np.array([0.1, 0.1, 0.1])
        rgb2 = np.array([0.6, 0.6, 0.6])
        default_rgb = np.array([0.5, 0.5, 0.5])

        cur = world = self.add_link(None,  Pose(np.array([0, 0, 0]), PxIdentity), "world", contype=1, conaffinity=1) # root coordinates #free
        #table
        world.add_box_shape(Pose([0., 0.5, -0.325-0.01]), np.array((1., 1., 0.01)))
        world.add_box_visual(Pose([0., 0.5, -0.325-0.01]), size=np.array((1., 1., 0.01)), color=np.array([0.5, 0.5, 0.5]))


        tmp_contype = 1

        cur = r_shoulder_pan_link = self.my_add_link(world, ([0, -0.6, 0], PxIdentity), ([0, 0, 0], x2z),
                                               "r_shoulder_pan_link", "r_shoulder_pan_joint", [-2.2854, 1.714602], contype=0, conaffinity=0, damping=1.)
        self.add_sphere(cur, np.array([-0.06, 0.05, 0.2]), PxIdentity, 0.05, rgb2, "e1")
        self.add_sphere(cur, np.array([0.06, 0.05, 0.2]), PxIdentity, 0.05, rgb2, "e2")
        self.add_sphere(cur, np.array([-0.06, 0.09, 0.2]), PxIdentity, 0.03, rgb, "e1p")
        self.add_sphere(cur, np.array([0.06, 0.09, 0.2]), PxIdentity, 0.03, rgb, "e2p")
        self.fromto(cur, "0 0 -0.4 0 0 0.2", 0.1, default_rgb, "sp")


        cur = r_shoulder_lift_link = self.my_add_link(cur, ([0.1, 0, 0], PxIdentity), ([0, 0, 0], x2y),
                                               "r_shoulder_lift_link", "r_shoulder_lift_joint", [-0.5236, 1.3963], damping=1., contype=0, conaffinity=0)
        self.fromto(cur, "0 -0.1 0 0 0.1 0", 0.1, default_rgb, "sl")

        cur = r_upper_arm_roll_link = self.my_add_link(cur, ([0, 0, 0], PxIdentity), ([0, 0, 0], PxIdentity),
                                               "r_upper_arm_roll_link", "r_upper_arm_roll_joint", [-1.5, 1.7], contype=0, conaffinity=0, damping=0.1)

        self.fromto(cur, "-0.1 0 0 0.1 0 0", 0.02, default_rgb, "uar")
        self.fromto(cur, "0 0 0 0.4 0 0", 0.06, default_rgb, "ua")

        cur = r_elbow_flex_link = self.my_add_link(cur, ([0.4, 0, 0], PxIdentity), ([0, 0, 0], x2y),
                                                 "r_elbow_flex_link", "r_elbow_flex_joint", [-2.3213, 0], contype=0, conaffinity=0, damping=0.1)
        self.fromto(cur, "0 -0.02 0 0.0 0.02 0", 0.06, default_rgb, "ef")

        cur = r_forearm_roll_link = self.my_add_link(cur, ([0, 0, 0], PxIdentity), ([0, 0, 0], PxIdentity),
                                             "r_forearm_roll_link", "r_forearm_roll_joint", [-1.5, 1.5], contype=0, conaffinity=0, damping=0.1)
        self.fromto(cur, "-0.1 0 0 0.1 0 0", 0.02, default_rgb, "fr")
        self.fromto(cur, "0 0 0 0.291 0 0", 0.05, default_rgb, "fa")


        cur = r_wrist_flex_link = self.my_add_link(cur, ([0.321, 0, 0], PxIdentity), ([0, 0, 0], x2y),
                                               "r_wrist_flex_link", "r_wrist_flex_joint", [-1.094, 0.], contype=0, conaffinity=0, damping=0.1)
        self.fromto(cur, "0 -0.02 0 0 0.02 0", 0.01, rgb, "wf")

        cur = r_wrist_roll_link = self.my_add_link(cur, ([0., 0, 0], PxIdentity), ([0, 0, 0], PxIdentity),
                                             "r_wrist_roll_link", "r_wrist_roll_joint", [-1.5, 1.5], contype=1, conaffinity=1, damping=0.1)

        self.add_sphere(cur, np.array([0.1, -0.1, 0.]), PxIdentity, 0.01, default_rgb, "tip_arml")
        self.add_sphere(cur, np.array([0.1, 0.1, 0.]), PxIdentity, 0.01, default_rgb, "tip_armr")

        self.fromto(cur, "0 -0.1 0. 0.0 +0.1 0", 0.02, default_rgb, "hand1")
        self.fromto(cur, "0 -0.1 0. 0.1 -0.1 0", 0.02, default_rgb, "hand2")
        self.fromto(cur, "0 +0.1 0. 0.1 +0.1 0", 0.02, default_rgb, "hand3")



        obj_slidey = self.my_add_link(world, ([0.45, -0.05, -0.225+0.01], PxIdentity), ((0, 0, 0), x2y), "obj_slidey", "obj_slidey", [-10.3213, 10.3], damping=10, type='slider', contype=0, conaffinity=0)
        obj_slidey.set_mass_and_inertia(1e-6, Pose(), [1, 1, 1])
        obj = self.my_add_link(obj_slidey, ([0.0, 0.0, 0.0], PxIdentity), ((0, 0, 0), PxIdentity), "object", "obj_slidex", [-10.3213, 10.3], damping=10, type='slider', contype=1, conaffinity=1)
        self.add_capsule(obj, (0, 0, 0), x2z, 0.05, 0.05, (1, 1, 1), "object", density=567)
        #self.add_box(obj, (0, 0, 0), x2z, (0.05, 0.05, 0.05), (1, 1, 1), "object", density=0.0001)

        goal_slidey = self.my_add_link(world, ([0.45, -0.05, -0.323], PxIdentity), ((0, 0, 0), x2y), "goal_slidey", "goal_slidey", [-10.3213, 10.3], damping=0.5, type='slider')
        goal = self.my_add_link(goal_slidey, ([0.0, 0.0, 0.0], PxIdentity), ((0, 0, 0), PxIdentity), "goal", "goal_sliderx", [-10.3213, 10.3], damping=0.5, type='slider', contype=0, conaffinity=0)
        self.add_box(goal, (0, 0, 0), PxIdentity, np.array((0.08, 0.08, 0.001)), (1, 0, 0), "goal", density=0.0001)



        wrapper = builder.build(True) #fix base = True
        #ground = self.sim.add_ground(-1)
        wrapper.set_root_pose(Pose([0., 0., 0.]))

        limit = 2.
        self.limit = limit
        self.add_force_actuator("r_shoulder_pan_joint", -limit, limit)
        self.add_force_actuator("r_shoulder_lift_joint", -limit, limit)
        self.add_force_actuator("r_upper_arm_roll_joint", -limit, limit)
        self.add_force_actuator("r_elbow_flex_joint", -limit, limit)
        self.add_force_actuator("r_forearm_roll_joint", -limit, limit)
        self.add_force_actuator("r_wrist_flex_joint", -limit, limit)
        self.add_force_actuator("r_wrist_roll_joint", -limit, limit)

        self._link_dict = {i.name: i for i in wrapper.get_links()}
        for _, link in self._link_dict.items():
            #link.set_inertia(np.array(link.inertia) + 0.04)
            link.set_mass_and_inertia(link.get_mass(), np.array(link.inertia) + 0.04)
        return wrapper, None

    def get_body_com(self, body_name):
        if body_name == 'tips_arm':
            return self._link_dict['r_wrist_roll_link'].pose.p
        else:
            return self._link_dict[body_name].pose.p

    def step(self, a):
        a = np.clip(a, -self.limit, self.limit)

        vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        reward_near = - np.linalg.norm(vec_1)
        reward_dist = - np.linalg.norm(vec_2)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist,
                reward_ctrl=reward_ctrl)

    def do_simulation(self, a, n_frames):
        qf = np.zeros((self._dof), np.float32)
        qf[self.actor_idx] = a
        for _ in range(n_frames):
            #self._link_dict['object'].add_force_torque([-0.1, -0.1, 0], [0, 0, 0])
            self.model.set_qf(qf)
            self.sim.step()


    def reset_model(self):
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
            self.get_body_com("tips_arm"),
            self.get_body_com("object"),
            self.get_body_com("goal"),
        ])