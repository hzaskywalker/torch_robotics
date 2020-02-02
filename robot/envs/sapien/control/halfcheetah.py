from gym import utils
from . import sapien_env
import numpy as np

from .sapien_env import sapien_core, Pose


class HalfCheetahEnv(sapien_env.SapienEnv, utils.EzPickle):
    def __init__(self):
        sapien_env.SapienEnv.__init__(self, 5, 0.01)
        utils.EzPickle.__init__(self)


    def build_render(self):
        self.sim.set_ambient_light([.4, .4, .4])
        self.sim.set_shadow_light([1, -1, -1], [.5, .5, .5])
        self.sim.add_point_light([2, 2, 2], [1, 1, 1])
        self.sim.add_point_light([2, -2, 2], [1, 1, 1])
        self.sim.add_point_light([-2, 0, 2], [1, 1, 1])

        self._renderer.set_camera_position(0, -8, 4)
        self._renderer.set_camera_rotation(np.pi/2, -0.5)


    def build_model(self):
        builder = self.builder
        PxIdentity = np.array([1, 0, 0, 0])
        x2z = np.array([0.7071068, 0, -0.7071068, 0])
        x2y = np.array([0.7071068, 0, 0, 0.7071068])
        #density = 5

        root1 = self.add_link(None,  Pose(np.array([0, 0, 0]), PxIdentity), "root1")
        root2 = self.my_add_link(root1, ((0, 0, 0.7), PxIdentity), ((0, 0, 0), PxIdentity), name='rootx', damping=0, stiffness=0, type='slider', range=[-np.inf, np.inf], joint_name='rootx')
        root3 = self.my_add_link(root2, ((0, 0, 0), PxIdentity), ((0, 0, 0), x2z), name='rootz', damping=0, stiffness=0, type='slider', range=[-np.inf, np.inf], joint_name='rootz')
        torso = self.my_add_link(root3, ((0, 0, 0), PxIdentity), ((0, 0, 0), x2y), name='torso', damping=0, stiffness=0, type='hinge', range=[-np.inf, np.inf], joint_name='rooty')

        self.fromto(torso, "-0.5 0 0 0.5 0 0", 0.046, np.array([0.4, 0.6, 0.8]), "torso")
        self.add_capsule(torso,  np.array([0.6, 0, 0.1]),
                                        np.array([0.939236, 0.000000, -0.343272, 0.000000]), 0.046, 0.15, np.array([0.4, 0.6, 0.8]), "head")

        bthigh = self.add_link(torso, Pose(np.array([0, 0, 0]), PxIdentity), "bthigh", "bthigh",
                                 sapien_core.ArticulationJointType.REVOLUTE, np.array([[-0.52, 1.05]]),
                                 Pose(np.array([-0.5, 0, 0]), np.array([0.7071068, 0, 0, 0.7071068])),
                                 Pose(np.array([0, 0, 0]), np.array([0.7071068, 0, 0, 0.7071068])))

        self.add_capsule(bthigh,  np.array([0.1, 0, -0.13]),
                                            np.array([-0.897736,-0.000000,-0.440535,0.000000]), 0.046, 0.145,
                                            np.array([1, 1, 1]), "bthigh")

        bshin = self.add_link(bthigh, Pose(np.array([0, 0, 0]), PxIdentity), "bshin", "bshin",
                                 sapien_core.ArticulationJointType.REVOLUTE, np.array([[-0.785, 0.785]]),
                                 Pose(np.array([0.16, 0, -.25]), np.array([ 0.7071068, 0, 0, 0.7071068])),
                                 Pose(np.array([0, 0, 0]), np.array([0.7071068, 0, 0, 0.7071068])))

        self.add_capsule(bshin,  np.array([-0.14, 0, -0.07]),
                                        np.array([-0.227590, 0.000000, -0.973757, 0.000000]), 0.046, 0.15,
                                            np.array([0.9, 0.6, 0.6]), "bshin")

        bfoot = self.add_link(bshin, Pose(np.array([0, 0, 0]), PxIdentity), "bfoot", "bfoot",
                                 sapien_core.ArticulationJointType.REVOLUTE, np.array([[-0.4, 0.785]]),
                                 Pose(np.array([-0.28, 0, -.14]), np.array([ 0.7071068, 0, 0, 0.7071068])),
                                 Pose(np.array([0, 0, 0]), np.array([0.7071068, 0, 0, 0.7071068]))) 
        self.add_capsule(bfoot, np.array([0.03, 0, -0.097]),
                                        np.array([0.605503, 0.000000, -0.795843, 0.000000]), 0.046, 0.094,
                                            np.array([0.9, 0.6, 0.6]), "bfoot")


        fthigh = self.add_link(torso, Pose(np.array([0, 0, 0]), PxIdentity), "fthigh", "fthigh",
                                 sapien_core.ArticulationJointType.REVOLUTE, np.array([[-1, 0.7]]),
                                 Pose(np.array([0.5, 0, 0]), np.array([0.7071068, 0, 0, 0.7071068])),
                                 Pose(np.array([0, 0, 0]), np.array([0.7071068, 0, 0, 0.7071068]))) 
        self.add_capsule(fthigh,  np.array([-0.07, 0, -0.12]),
                                            np.array([0.865124, 0.000000, -0.501557, 0.000000]), 0.046, 0.133,
                                            np.array([1, 1, 1]), "fthigh")

        fshin = self.add_link(fthigh, Pose(np.array([0, 0, 0]), PxIdentity), "fshin", "fshin",
                                 sapien_core.ArticulationJointType.REVOLUTE, np.array([[-1.2, .87]]),
                                 Pose(np.array([-.14, 0, -.24]), np.array([ 0.7071068, 0, 0, 0.7071068])),
                                 Pose(np.array([0, 0, 0]), np.array([0.7071068, 0, 0, 0.7071068]))) 
        self.add_capsule(fshin, np.array([.065, 0, -.09]),
                                        np.array([0.466561, 0.000000, -0.884489, 0.000000]), 0.046, 0.106,
                                            np.array([0.9, 0.6, 0.6]), "fshin")

        ffoot = self.add_link(fshin, Pose(np.array([0, 0, 0]), PxIdentity), "ffoot", "ffoot",
                                 sapien_core.ArticulationJointType.REVOLUTE, np.array([[-0.5, 0.5]]),
                                 Pose(np.array([0.13, 0, -.18]), np.array([ 0.7071068, 0, 0, 0.7071068])),
                                 Pose(np.array([0, 0, 0]), np.array([0.7071068, 0, 0, 0.7071068]))) 
        self.add_capsule(ffoot, np.array([0.045, 0, -0.07]),
                                        np.array([0.466561, 0.000000, -0.884489, 0.000000]), 0.046, 0.07,
                                            np.array([0.9, 0.6, 0.6]), "ffoot")

        wrapper = builder.build(True)
        self.add_force_actuator("bthigh", -1, 1)
        self.add_force_actuator("bshin", -1, 1)
        self.add_force_actuator("bfoot", -1, 1)
        self.add_force_actuator("fthigh", -1, 1)
        self.add_force_actuator("fshin", -1, 1)
        self.add_force_actuator("ffoot", -1, 1)

        self.torso = wrapper.get_links()[0]
        wrapper.set_root_pose(Pose([0, 0, 0]))
        ground = self.sim.add_ground(0)
        return wrapper, None

    def step(self, a):
        dd = np.array([120 * 0.84, 90 * 0.800, 60 * 0.766, 120 * 0.98, 60 * 0.875, 30 * 0.947])
        xposbefore = self.get_qpos()[0]
        a = np.clip(a, -1, 1)
        self.do_simulation(a * dd, self.frame_skip)
        xposafter = self.get_qpos()[0]

        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(a).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.model.get_qpos().flat[1:],
            self.model.get_qvel().flat,
        ])


    def render(self, mode='human', width=500, height=500):
        if '_renderer' in self.__dict__:
            self._renderer.set_camera_position(self.get_qpos()[0], -8, 4)
        return self._get_viewer(mode).render()

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qpos += self.np_random.uniform(size=9, low=-.1, high=.1)
        qvel = self.init_qvel.copy()
        qvel += self.np_random.randn(9) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

