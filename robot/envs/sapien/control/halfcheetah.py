from gym import utils
from . import sapien_env
import numpy as np

try:
    import sapyen as sapien_core
    from sapyen import Pose
except ModuleNotFoundError:
    import sapien.core as sapien_core
    from sapien.core import Pose


class HalfCheetahEnv(sapien_env.SapienEnv, utils.EzPickle):
    def __init__(self):
        sapien_env.SapienEnv.__init__(self, 5, 0.01)
        utils.EzPickle.__init__(self)


    def build_render(self):
        renderer = sapien_core.OptifuserRenderer()

        renderer.set_ambient_light([.4, .4, .4])
        renderer.set_shadow_light([1, -1, -1], [.5, .5, .5])
        renderer.add_point_light([2, 2, 2], [1, 1, 1])
        renderer.add_point_light([2, -2, 2], [1, 1, 1])
        renderer.add_point_light([-2, 0, 2], [1, 1, 1])

        renderer.cam.set_position(np.array([0, -10, 10]))
        renderer.cam.rotate_yaw_pitch(0, -0.5)
        return renderer


    def build_model(self):
        builder = self.sim.create_articulation_builder()
        PxIdentity = np.array([1, 0, 0, 0])
        x2z = np.array([0.7071068, 0, -0.7071068, 0])
        x2y = np.array([0.7071068, 0, 0, 0.7071068])
        #density = 5

        root1 = builder.add_link(None,  Pose(np.array([0, 0, 0]), PxIdentity), "root1")
        root2 = builder.add_link(root1, Pose(np.array([0, 0, 0]), PxIdentity), "root2", "fake1",
                                sapien_core.PxArticulationJointType.PRISMATIC, np.array([[-np.inf, np.inf]]),
                                Pose(np.array([0, 0, 0]), x2z), Pose(np.array([0, 0, 0]), x2z))
        root3 = builder.add_link(root2, Pose(np.array([0, 0, 0]), PxIdentity), "root3", "fake2",
                                sapien_core.PxArticulationJointType.PRISMATIC, np.array([[-np.inf, np.inf]]))
        torso = builder.add_link(root3, Pose(np.array([0, 0, 0]), PxIdentity), "torso", "torso",
                                sapien_core.PxArticulationJointType.REVOLUTE, np.array([[-np.inf, np.inf]]),
                                Pose(np.array([0, 0, 0]), x2y), Pose(np.array([0, 0, 0]), x2y))                     

        self.add_capsule(builder, torso, np.array([0, 0, 0]), PxIdentity, 0.046, 0.5,
                                          np.array([1, 1, 1]), "torso")
        
        self.add_capsule(builder, torso,  np.array([0.6, 0, 0.1]),
                                        np.array([0.939236, 0.000000, -0.343272, 0.000000]), 0.046, 0.15,
                                          np.array([1, 1, 1]), "head")

        bthigh = builder.add_link(torso, Pose(np.array([0, 0, 0]), PxIdentity), "bthigh", "bthigh",
                                 sapien_core.PxArticulationJointType.REVOLUTE, np.array([[-0.52, 1.05]]),
                                 Pose(np.array([-0.5, 0, 0]), np.array([0.7071068, 0, 0, 0.7071068])),
                                 Pose(np.array([0, 0, 0]), np.array([0.7071068, 0, 0, 0.7071068])))

        self.add_capsule(builder, bthigh,  np.array([0.1, 0, -0.13]),
                                            np.array([-0.897736,-0.000000,-0.440535,0.000000]), 0.046, 0.145,
                                            np.array([1, 1, 1]), "bthigh")

        bshin = builder.add_link(bthigh, Pose(np.array([0, 0, 0]), PxIdentity), "bshin", "bshin",
                                 sapien_core.PxArticulationJointType.REVOLUTE, np.array([[-0.785, 0.785]]),
                                 Pose(np.array([0.16, 0, -.25]), np.array([ 0.7071068, 0, 0, 0.7071068])),
                                 Pose(np.array([0, 0, 0]), np.array([0.7071068, 0, 0, 0.7071068])))

        self.add_capsule(builder, bshin,  np.array([-0.14, 0, -0.07]),
                                        np.array([-0.227590, 0.000000, -0.973757, 0.000000]), 0.046, 0.15,
                                            np.array([0.9, 0.6, 0.6]), "bshin")

        bfoot = builder.add_link(bshin, Pose(np.array([0, 0, 0]), PxIdentity), "bfoot", "bfoot",
                                 sapien_core.PxArticulationJointType.REVOLUTE, np.array([[-0.4, 0.785]]),
                                 Pose(np.array([-0.28, 0, -.14]), np.array([ 0.7071068, 0, 0, 0.7071068])),
                                 Pose(np.array([0, 0, 0]), np.array([0.7071068, 0, 0, 0.7071068]))) 
        self.add_capsule(builder, bfoot, np.array([0.03, 0, -0.097]),
                                        np.array([0.605503, 0.000000, -0.795843, 0.000000]), 0.046, 0.094,
                                            np.array([0.9, 0.6, 0.6]), "bfoot")


        fthigh = builder.add_link(torso, Pose(np.array([0, 0, 0]), PxIdentity), "fthigh", "fthigh",
                                 sapien_core.PxArticulationJointType.REVOLUTE, np.array([[-1, 0.7]]),
                                 Pose(np.array([0.5, 0, 0]), np.array([0.7071068, 0, 0, 0.7071068])),
                                 Pose(np.array([0, 0, 0]), np.array([0.7071068, 0, 0, 0.7071068]))) 
        self.add_capsule(builder, fthigh,  np.array([-0.07, 0, -0.12]),
                                            np.array([0.865124, 0.000000, -0.501557, 0.000000]), 0.046, 0.133,
                                            np.array([1, 1, 1]), "fthigh")

        fshin = builder.add_link(fthigh, Pose(np.array([0, 0, 0]), PxIdentity), "fshin", "fshin",
                                 sapien_core.PxArticulationJointType.REVOLUTE, np.array([[-1.2, .87]]),
                                 Pose(np.array([-.14, 0, -.24]), np.array([ 0.7071068, 0, 0, 0.7071068])),
                                 Pose(np.array([0, 0, 0]), np.array([0.7071068, 0, 0, 0.7071068]))) 
        self.add_capsule(builder, fshin, np.array([.065, 0, -.09]),
                                        np.array([0.466561, 0.000000, -0.884489, 0.000000]), 0.046, 0.106,
                                            np.array([0.9, 0.6, 0.6]), "fshin")

        ffoot = builder.add_link(fshin, Pose(np.array([0, 0, 0]), PxIdentity), "ffoot", "ffoot",
                                 sapien_core.PxArticulationJointType.REVOLUTE, np.array([[-0.5, 0.5]]),
                                 Pose(np.array([0.13, 0, -.18]), np.array([ 0.7071068, 0, 0, 0.7071068])),
                                 Pose(np.array([0, 0, 0]), np.array([0.7071068, 0, 0, 0.7071068]))) 
        self.add_capsule(builder, ffoot, np.array([0.045, 0, -0.07]),
                                        np.array([0.466561, 0.000000, -0.884489, 0.000000]), 0.046, 0.07,
                                            np.array([0.9, 0.6, 0.6]), "ffoot")

        wrapper = builder.build(True)
        wrapper.add_force_actuator("bthigh", -120, 120)
        wrapper.add_force_actuator("bshin", -90, 90)
        wrapper.add_force_actuator("bfoot", -60, 60)
        wrapper.add_force_actuator("fthigh", -120, 120)
        wrapper.add_force_actuator("fshin", -60, 60)
        wrapper.add_force_actuator("ffoot", -30, 30)

        #self.root_link = torso
        #self.init_root_pose_p = self.root_link.get_global_pose().p
        #self.init_root_pose_q = self.root_link.get_global_pose().q
        #self.add_object(torso)
        ground = self.sim.add_ground(-1)
        return wrapper, torso

    def step(self, a):
        xposbefore = self.root.get_global_pose().p[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.root.get_global_pose().p[0]

        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = -.1 * np.square(a / np.array([120, 90, 60, 120, 60, 30])).sum()
        reward = forward_reward + ctrl_cost 
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(
            reward_run=forward_reward,
            reward_ctrl=-ctrl_cost)

    def _get_obs(self):
        return np.concatenate([
            self.model.get_qpos().flat[1:],
            self.model.get_qvel().flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qpos[7:] += self.np_random.uniform(size=9, low=-.1, high=.1)
        qvel = self.init_qvel.copy()
        qvel[6:] += self.np_random.randn(9) * .1
        #self.model.set_qpos(qpos)
        #self.model.set_qvel(qvel)
        #root_pose_p = self.init_root_pose_p #+ self.np_random.uniform(size=3, low=-.1, high=.1)
        #root_pose_q = self.init_root_pose_q #+ self.np_random.uniform(size=4, low=-.1, high=.1)
        #self.model.set_root_pose(root_pose_p, root_pose_q)
        self.set_state(qpos, qvel)
        # TODO: reset root_velocity
        return self._get_obs()

