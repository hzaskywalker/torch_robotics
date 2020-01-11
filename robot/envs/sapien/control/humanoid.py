import numpy as np
from .sapien_env import Pose, sapien_core, SapienEnv
from gym import utils
from transforms3d.quaternions import qmult, rotate_vector, axangle2quat

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class HumanoidEnv(SapienEnv, utils.EzPickle):
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

        renderer.cam.set_position(np.array([0, -4, 4]))
        renderer.cam.rotate_yaw_pitch(0, -0.5)
        return renderer

    def build_model(self):
        def parent_pose(xpos, xquat, ypos, yquat):
            pos = rotate_vector(ypos, xquat) + xpos
            quat = qmult(xquat, yquat)
            return Pose(pos, quat)

        def vec2pose(vec):
            l = np.linalg.norm(vec)
            a, b = np.array([1, 0, 0]), vec/l
            # find quat such that qmult(quat, [1, 0, 0]) = vec
            v = np.cross(a, b) #rotation along v
            theta = np.arccos(np.dot(a, b))
            pose = axangle2quat(v, theta)
            assert np.linalg.norm(rotate_vector(np.array([1, 0, 0]), pose) - b) < 1e-5
            return l, pose

        builder = self.sim.create_articulation_builder()
        PxIdentity = np.array([1, 0, 0, 0])
        x2y = np.array([0.7071068, 0, 0, 0.7071068])
        x2z = np.array([0.7071068, 0, 0.7071068, 0])
        default_rgb = np.array([0.8, 0.6, 0.4])

        torso = builder.add_link(None,  Pose(np.array([0, 0, 1.4]), PxIdentity), "torso") # root coordinates #free
        self.add_capsule(builder, torso, np.array([0, 0., 0]), x2y, 0.07, 0.07, default_rgb, "torso1")
        self.add_sphere(builder, torso, np.array([0, 0., 0.19]), x2y, 0.09, default_rgb, "head")
        self.add_capsule(builder, torso, np.array([-0.01, 0, -0.12]), x2y, 0.06, 0.06, default_rgb, "uwaist")


        # TODO: armature, damping, stiffness
        lwist_pos = np.array([-0.01, 0, -0.260]), np.array([1., 0, -0.002, 0])
        joint_pos = parent_pose(*lwist_pos, np.array([0., 0., 0.065]), x2z)
        lwaist_fake = builder.add_link(torso,  Pose(np.array([0, 0, 0.]), PxIdentity), "lwaist_fake", "abdomen_z",
                                       sapien_core.PxArticulationJointType.REVOLUTE, np.array([[np.radians(-45), np.radians(45)]]),
                                       joint_pos, joint_pos,
                                       ) # end in torso's framework
        joint_pos = parent_pose(*lwist_pos, np.array([0., 0., 0.065]), x2y)
        lwaist = builder.add_link(lwaist_fake,  Pose(np.array([0, 0, 0.]), PxIdentity), "lwaist", "abdomen_y",
                                  sapien_core.PxArticulationJointType.REVOLUTE, np.array([[np.radians(-75), np.radians(30)]]),
                                  joint_pos, Pose(np.array([0., 0., 0.065]), x2y)
                                  )
        # NOTE: the default of mujoco is rot in z direction, we need z2
        self.add_capsule(builder, lwaist, np.array([0., 0, 0.]), x2y, 0.06, 0.06, default_rgb, "lwaist")


        joint_pos = np.array([0., 0., 0.1]), PxIdentity
        pelvis = builder.add_link(lwaist,  Pose(np.array([0, 0, 0.]), PxIdentity), "pelvis", "abdomen_x",
                                  sapien_core.PxArticulationJointType.REVOLUTE, np.array([[np.radians(-35), np.radians(35)]]),
                                  parent_pose(np.array([0, 0, -0.165]), np.array([1., 0, -0.002, 0]), *joint_pos),
                                  Pose(*joint_pos)
                              )
        self.add_capsule(builder, pelvis, np.array([-0.02, 0., 0.]), x2y, 0.09, 0.07, default_rgb, "butt")


        for dir in ['right', 'left']:
            if dir == 'right':
                local_pos = Pose(np.array([0, -0.1, -0.04]), PxIdentity)
            else:
                local_pos = Pose(np.array([0, 0.1, -0.04]), PxIdentity)
            hip_x = builder.add_link(pelvis, Pose(np.array([0, 0, 0.]), PxIdentity), f"{dir}_hip_x", f"{dir}_hip_x",
                                     sapien_core.PxArticulationJointType.REVOLUTE,
                                     np.array([[np.radians(-25), np.radians(5)]]),
                                     local_pos, Pose(np.array([0, 0, 0]), PxIdentity),
                                     )  # end in local pos
            hip_z_pos = Pose(np.array([0, 0, 0]), PxIdentity)
            hip_z = builder.add_link(hip_x, Pose(np.array([0, 0, 0.]), PxIdentity), f"{dir}_hip_z", f"{dir}_hip_z",
                                     sapien_core.PxArticulationJointType.REVOLUTE,
                                     np.array([[np.radians(-60), np.radians(35)]]),
                                     hip_z_pos, hip_z_pos)  # end in local pos

            hip_y_pos = Pose(np.array([0, 0, 0]), x2y)
            thigh = builder.add_link(hip_z, Pose(np.array([0, 0, 0.]), PxIdentity), f"{dir}_thigh", f"{dir}_hip_y",
                                     sapien_core.PxArticulationJointType.REVOLUTE,
                                     np.array([[np.radians(-110), np.radians(20)]]),
                                     hip_y_pos, hip_y_pos)  # end in local pos

            if dir == 'right':
                vec = np.array([0, 0.01, -0.34])/2
            else:
                vec = np.array([0, -0.01, -0.34])/2

            l, quat = vec2pose(vec)
            # TODO: there will be no collision between, ideally it should be thigh, but not hip_x
            th = hip_x

            self.add_capsule(builder, th, vec, quat, 0.06, l, default_rgb, f"{dir}_thigh1")


            if dir == 'right':
                shin_pos = np.array([0, 0.01, -0.403]), PxIdentity
            else:
                shin_pos = np.array([0, -0.01, -0.403]), PxIdentity
            local_pos = np.array([0, 0, 0.02]), np.array([0.7071068, 0, 0, -0.7071068]) #x2-y
            shin = builder.add_link(th, Pose(np.array([0, 0, 0.]), PxIdentity), f"{dir}_shin", f"{dir}_knee",
                                     sapien_core.PxArticulationJointType.REVOLUTE,
                                     np.array([[np.radians(-160), np.radians(-2)]]),
                                     parent_pose(*shin_pos, *local_pos), Pose(*local_pos),
                                     )  # end in local pos
            self.add_capsule(builder, shin, np.array([0, 0, -0.15]), np.array([0.7071068, 0, -0.7071068, 0]), #x2-z
                             0.049, 0.15, default_rgb, f"{dir}_shin")
            self.add_sphere(builder, shin, np.array([0, 0, -0.35]), PxIdentity, #x2-z
                             0.075, default_rgb, f"{dir}_foot")


            # arm
            if dir == 'right':
                arm_pose = np.array([0, -0.17, 0.06]), PxIdentity
            else:
                arm_pose = np.array([0, 0.17, 0.06]), PxIdentity

            local_pos = np.array([0, 0, 0]), vec2pose(np.array([2, 1, 1]))[1]
            shoulder_1 = builder.add_link(torso, Pose(np.array([0, 0, 0.]), PxIdentity), f"{dir}_shoulder1", f"{dir}_shoulder1",
                                     sapien_core.PxArticulationJointType.REVOLUTE,
                                     np.array([[np.radians(-85), np.radians(60)]]),
                                     parent_pose(*arm_pose, *local_pos), Pose(*local_pos),
                                     )  # end in local pos

            local_pos = np.array([0, 0, 0]), vec2pose(np.array([0, -1, 1]))[1]
            upper_arm = builder.add_link(shoulder_1, Pose(np.array([0, 0, 0.]), PxIdentity), f"{dir}_upper_arm", f"{dir}_shoulder2",
                                          sapien_core.PxArticulationJointType.REVOLUTE,
                                          np.array([[np.radians(-85), np.radians(60)]]),
                                          Pose(*local_pos), Pose(*local_pos),
                                          )  # end in local pos
            #TODO: should be upper_arm actaully
            #arm = upper_arm
            arm = shoulder_1

            if dir == 'right':
                vec = np.array([0.16, -0.16, -0.16])/2
            else:
                vec = np.array([0.16, 0.16, -0.16])/2

            l, quat = vec2pose(vec)
            self.add_capsule(builder, arm, vec, quat, 0.04, 0.16, default_rgb, f"{dir}_thigh1")



            # lower
            if dir == 'right':
                arm_pose = np.array([0.18, -0.18, -0.18]), PxIdentity
            else:
                arm_pose = np.array([0.18, 0.18, -0.18]), PxIdentity
            local_pos = np.array([0, 0, 0]), vec2pose(np.array([0, -1, -1]))[1]
            lower_arm = builder.add_link(arm, Pose(np.array([0, 0, 0.]), PxIdentity), f"{dir}_lower_arm", f"{dir}_elbow",
                                          sapien_core.PxArticulationJointType.REVOLUTE,
                                          np.array([[np.radians(-90), np.radians(50)]]),
                                          parent_pose(*arm_pose, *local_pos), Pose(*local_pos),
                                      )  # end in local pos


            if dir == 'right':
                vec = np.array([0.01, 0.01, 0.01, 0.17, 0.17, 0.17])
            else:
                vec = np.array([0.01, -0.01, 0.01, 0.17, -0.17, 0.17])
            middle = (vec[3:] + vec[:3])/2

            l, quat = vec2pose((vec[3:] - vec[:3])/2)
            self.add_capsule(builder, lower_arm, middle, quat, 0.031, l, default_rgb, f"{dir}_lower_arm")

            pos = np.array([0.18, 0.18, 0.18]) if dir == 'right' else np.array([0.18, -0.18, 0.18])
            self.add_sphere(builder, lower_arm, pos, PxIdentity, 0.04, default_rgb, f"{dir}_hand")

        #TODO: tendon


        wrapper = builder.build(True) #fix base = True
        wrapper.add_force_actuator("abdomen_z", -100, 100)
        wrapper.add_force_actuator("abdomen_y", -100, 100)
        wrapper.add_force_actuator("abdomen_x", -100, 100)
        for i in ['left', 'right']:
            wrapper.add_force_actuator(f"{i}_hip_x", -100, 100)
            wrapper.add_force_actuator(f"{i}_hip_y", -100, 100)
            wrapper.add_force_actuator(f"{i}_hip_z", -300, 300)
            wrapper.add_force_actuator(f"{i}_keen", -200, 200)
            wrapper.add_force_actuator(f"{i}_shouler1", -25, 25)
            wrapper.add_force_actuator(f"{i}_shouler2", -25, 25)
            wrapper.add_force_actuator(f"{i}_elbow", -25, 25)

        ground = self.sim.add_ground(-1)
        return wrapper, torso

    def _get_obs(self):
        return np.concatenate([self.get_qpos().flat[2:],
                               self.get_qvel().flat,
                               #data.cinert.flat,           TODO:com-based body inertia and mass n x 10
                               #data.cvel.flat,             TODO:com-based velocity n x 6

                               #data.qfrc_actuator.flat,    TODO:actuator force
                               #data.cfrc_ext.flat          external force on body
                               np.clip(self.model.get_cfrc_ext(), -1, 1).flat,  # 54->84 ??
                               ])

    def step(self, a):
        #TODO: pos_before = mass_center(self.model, self.sim)
        pos_before = 0
        self.do_simulation(a, self.frame_skip)
        #pos_after = mass_center(self.model, self.sim)
        pos_after = 0
        alive_bonus = 5.0
        data = self.model
        lin_vel_cost = 1.25 * (pos_after - pos_before) / self.dt
        #TODO:quad_ctrl_cost = 0.1 * np.square(data.get_ctrl()).sum()
        quad_ctrl_cost = 0.1 * np.square(a).sum()
        quad_impact_cost = .5e-6 * np.square(data.get_cfrc_ext()).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.get_qpos()
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus, reward_impact=-quad_impact_cost)

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=len(self.init_qpos)),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=len(self.init_qvel),)
        )
        return self._get_obs()

    #def viewer_setup(self):
    #    self.viewer.cam.trackbodyid = 1
    #    self.viewer.cam.distance = self.model.stat.extent * 1.0
    #    self.viewer.cam.lookat[2] = 2.0
    #    self.viewer.cam.elevation = -20