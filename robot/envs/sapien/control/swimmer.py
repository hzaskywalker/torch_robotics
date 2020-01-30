from gym import utils, spaces
# from gym.envs.mujoco import mujoco_env
from .sapien_env import SapienEnv, Pose
import numpy as np
import transforms3d

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

        self._renderer.set_camera_position(0, -5, 5)
        self._renderer.set_camera_rotation(np.pi/2, -0.5)
        return self._renderer

    def build_model(self):
        builder = self.builder
        PxIdentity = np.array([1, 0, 0, 0])

        x2y = np.array([0.7071068, 0, 0, 0.7071068])
        x2z = np.array([0.7071068, 0, 0.7071068, 0])


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
        mid = self.my_add_link(torso, ([0.5, 0, 0], [1, 0, 0, 0]), ([0, 0, 0], x2z), "mid", "rot2", range, damping=0.)
        self.fromto(mid, "0 0 0 -1 0 0", 0.1, np.array([0., 1., 0]), "mid", density=density)

        back = self.my_add_link(mid, ([-1., 0, 0], [1, 0, 0, 0]), ([0, 0, 0], x2z), "back", "rot3", range, damping=0.)
        self.fromto(back, "0 0 0 -1 0 0", 0.1, np.array([0., 0., 1.]), "back", density=density)

        wrapper = builder.build(True) #fix base = True
        self.add_force_actuator("rot2", -1, 1)
        self.add_force_actuator("rot3", -1, 1)

        self._viscosity_links = [i for i in wrapper.get_links() if i.name in ['torso', 'mid', 'back']]
        self.sim.add_ground(-1)
        return wrapper, None

    def get_position(self, force, torque):
        # return r x f = torque
        l = np.dot(force, force)
        if np.abs(l) < 1e-6:
            r = np.array((0, 0, 0))
        else:
            r = np.cross(force, torque) / l
        return r

    def do_simulation(self, a, n_frames):
        viscosity = 0.1
        density = 4000
        #density *= 10
        #viscosity *= 10
        s = (0.8, 0.2, 0.2) # 1 x 0.1 x 0.1
        d = np.mean(s)
        for _ in range(n_frames):
            for link in self._viscosity_links:
                pose = link.pose

                pose = Pose(pose.p + link.cmass_local_pose.p, pose.q) # pose is the frame associated with center of mass
                #cmass = pose.p + transforms3d.quaternions.rotate_vector(cmass_local_pose.p, pose.q)

                inverse_q = transforms3d.quaternions.qinverse(pose.q)
                inverse_p = -transforms3d.quaternions.rotate_vector(pose.p, inverse_q)
                w_local = transforms3d.quaternions.rotate_vector(link.angular_velocity, inverse_q) # to center of the mass
                v_local = transforms3d.quaternions.rotate_vector(link.velocity, inverse_q) + np.cross(inverse_p, w_local)

                """
                #sanity check here
                bw = transforms3d.quaternions.rotate_vector(w_local, pose.q)
                bv = transforms3d.quaternions.rotate_vector(v_local, pose.q) + np.cross(pose.p, bw)

                assert np.linalg.norm(bw - link.angular_velocity) < 1e-6
                assert np.linalg.norm(bv - link.velocity) < 1e-6
                """

                def apply_local_force_torque(force, torque):
                    #print(force)
                    r = self.get_position(force, torque)
                    point = pose.p + transforms3d.quaternions.rotate_vector(r, pose.q)
                    # TODO: force only rotation?
                    force = transforms3d.quaternions.rotate_vector(force, pose.q)
                    #print(pose.q)
                    #print(force, link.velocity, v_local)
                    link.add_force_at_point(force, point)

                viscosity_force = -3 * viscosity * np.pi * d * v_local
                viscosity_torque = -viscosity * np.pi * d * d * d * w_local
                #viscosity_force = - v_local
                #viscosity_torque = - w_local
                apply_local_force_torque(viscosity_force, viscosity_torque)

                #if np.max(np.abs(v_local)) > 10:
                #    print(v_local, w_local)
                #    exit(0)

                density_force = -0.5 * density * np.array([s[1]*s[2], s[0]*s[2], s[0]*s[1]]) * np.abs(v_local) * v_local
                density_torque = -1/64 * density * np.array(s) * np.array((s[1]**4+s[2]**4, s[0]**4 + s[2]**4,
                                                                         s[0]**4+s[1] ** 4)) * np.abs(w_local) * w_local

                apply_local_force_torque(density_force, density_torque)

            qf = np.zeros((self._dof), np.float32)
            qf[self.actor_idx] = a
            self.model.set_qf(qf)
            self.sim.step()

    def step(self, a):
        ctrl_cost_coeff = 0.0001
        xposbefore = self.model.get_qpos()[0]
        self.do_simulation(a * np.array([150, 150]), self.frame_skip)
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
