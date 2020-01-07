from .sapien_env import sapien_core, Pose
from gym import utils
from . import sapien_env
import numpy as np

class CartpoleEnv(sapien_env.SapienEnv, utils.EzPickle):
    PENDULUM_LENGTH = 0.6

    def __init__(self):
        sapien_env.SapienEnv.__init__(self, 2, timestep=0.02)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        return np.concatenate([
            self.model.get_qpos().flat,
            self.model.get_qvel().flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + np.random.normal(0, 0.1, np.shape(self.init_qpos))
        qvel = self.init_qvel + np.random.normal(0, 0.1, np.shape(self.init_qvel))
        self.set_state(qpos, qvel)
        return self._get_obs()

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        cost_lscale = CartpoleEnv.PENDULUM_LENGTH
        reward = np.exp(
            -np.sum(np.square(self._get_ee_pos(ob) - np.array([0.0, CartpoleEnv.PENDULUM_LENGTH]))) / (cost_lscale ** 2)
        )
        reward -= 0.01 * np.sum(np.square(a))

        done = False
        return ob, reward, done, {}

    def build_render(self):
        renderer = sapien_core.OptifuserRenderer()

        renderer.set_ambient_light([.4, .4, .4])
        renderer.set_shadow_light([1, -1, -1], [.5, .5, .5])
        renderer.add_point_light([2, 2, 2], [1, 1, 1])
        renderer.add_point_light([2, -2, 2], [1, 1, 1])
        renderer.add_point_light([-2, 0, 2], [1, 1, 1])

        #renderer.cam.set_position(np.array([0, -1, 1]))
        #renderer.cam.rotate_yaw_pitch(0, -0.5)
        return renderer

    @staticmethod
    def _get_ee_pos(x):
        x0, theta = x[0], x[1]
        return np.array([
            x0 - CartpoleEnv.PENDULUM_LENGTH * np.sin(theta),
            -CartpoleEnv.PENDULUM_LENGTH * np.cos(theta)
        ])

    def build_model(self):
        builder = self.sim.create_articulation_builder()
        PxIdentity = np.array([1, 0, 0, 0]) # rotation

        x2z = np.array([0.7071068, 0, -0.7071068, 0])
        x2y = np.array([0.7071068, 0, 0, 0.7071068])

        root = builder.add_link(None,  Pose(np.array([0, 0, 0]), PxIdentity), "root") # world root

        cart = builder.add_link(root, Pose(np.array([0, 0, 0]), PxIdentity), "cart", "slider",
                                 sapien_core.PxArticulationJointType.REVOLUTE, np.array([[-2.5, 2.5]]),
                                 Pose(np.array([0, 0, 0]), x2z), Pose(np.array([0, 0, 0]), x2z))

        pole = builder.add_link(cart, Pose(np.array([0, 0, 0]), PxIdentity), "torso", "torso",
                                 sapien_core.PxArticulationJointType.REVOLUTE, np.array([[-np.pi, np.pi]]),
                                 Pose(np.array([0, 0, 0]), x2y), Pose(np.array([0, 0, 0]), x2y))

        self.add_capsule(builder, root, np.array([0, 0, 0]), np.array([0.707, 0, 0.707, 0]), 0.02, 3,
                         np.array([0.3, 0.3, 0.7]), "root_geom")

        self.add_capsule(builder, cart, np.array([0, 0, 0]), np.array([0.707, 0, 0.707, 0]), 0.1, 0.1,
                         np.array([1., 0., 0.]), "cart_geom")

        self.add_capsule(builder, pole, np.array([0, 0, 0]), np.array([0.707, 0, 0.707, 0]), 0.049, 0.3,
                         np.array([0, 0.7, 0.7]), "pole_geom")

        wrapper = builder.build(True)
        wrapper.add_force_actuator("slider", -100, 100)

        self.sim.add_ground(-1)
        return wrapper

