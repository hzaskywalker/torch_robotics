from .sapien_env import sapien_core, Pose
from gym import utils
from . import sapien_env
import numpy as np

class CartpoleEnv(sapien_env.SapienEnv, utils.EzPickle):
    def __init__(self):
        sapien_env.SapienEnv.__init__(self, 2, timestep=0.02)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        return np.concatenate([
            self.model.get_qpos().flat,
            self.model.get_qvel().flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-0.01, high=0.01, size=np.shape(self.init_qpos))
        qvel = self.init_qvel + self.np_random.uniform(low=-0.01, high=0.01, size=np.shape(self.init_qvel))
        self.set_state(qpos, qvel)
        return self._get_obs()

    def step(self, a):
        a = a
        reward = 1.0
        self.do_simulation(a * 108, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        done = not notdone

        return ob, reward, done, {}

    def build_render(self):
        self.sim.set_ambient_light([.4, .4, .4])
        self.sim.set_shadow_light([1, -1, -1], [.5, .5, .5])
        self.sim.add_point_light([2, 2, 2], [1, 1, 1])
        self.sim.add_point_light([2, -2, 2], [1, 1, 1])
        self.sim.add_point_light([-2, 0, 2], [1, 1, 1])

        self._renderer.set_camera_position(0, -3, 2)
        self._renderer.set_camera_rotation(1.57, -0.5)

    def build_model(self):
        builder = self.builder
        PxIdentity = np.array([1, 0, 0, 0]) # rotation
        x2y = np.array([0.7071068, 0, 0, 0.7071068])

        rail = self.add_link(None,  Pose(np.array([0, 0, 0]), PxIdentity), "rail") # world root
        self.add_capsule(rail, np.array([0, 0, 0]), np.array([1., 0, 0, 0]), 0.02, 1,
                         np.array([1., 0., 0.]), "rail", shape=False)


        cart = self.add_link(rail, Pose(np.array([0, 0, 0]), PxIdentity), "cart", "slider",
                                 sapien_core.ArticulationJointType.PRISMATIC, np.array([[-1., 1.]]),
                                 Pose(np.array([0, 0, 0]), PxIdentity), Pose(np.array([0, 0, 0]), PxIdentity), damping=True)
        self.add_capsule(cart, np.array([0, 0, 0]), np.array([1., 0, 0, 0]), 0.1, 0.1,
                         np.array([0., 1., 0.]), "cart")

        pole = self.my_add_link(cart, ((0, 0, 0), PxIdentity), ((0, 0, 0), x2y), 'pole', 'hinge', range=[-np.pi/2, np.pi/2], type='hinge', damping=True)
        self.fromto(pole, "0 0 0 0.001 0 0.6", size=0.049, rgb=np.array([0., 0.7, 0.7]), name='cpole')

        wrapper = builder.build(True)
        self.add_force_actuator("slider", -3, 3)
        self.sim.add_ground(-1)
        return wrapper, None
