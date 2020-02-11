import numpy as np
from robot.envs.sapien.control.sapien_env import SapienEnv, Pose, sapien_core
from gym import utils

class Ball(SapienEnv):
    def __init__(self):
        SapienEnv.__init__(self, 2, timestep=0.02)
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

        self._renderer.set_camera_position(5, -3, 2)
        self._renderer.set_camera_rotation(1.57, -0.5)

    def build_model(self):
        builder = self.builder
        PxIdentity = np.array([1, 0, 0, 0])  # rotation
        x2z = np.array([0.7071068, 0, -0.7071068, 0])
        x2y = np.array([0.7071068, 0, 0, 0.7071068])
        rail = self.add_link(None, Pose(np.array([0, 0, 0]), PxIdentity), "rail")  # world root
        #self.add_capsule(rail, np.array([0, 0, 0]), np.array([1., 0, 0, 0]), 0.02, 1,
        #                 np.array([1., 0., 0.]), "rail", shape=False)

        for i in range(10):
            root2 = self.my_add_link(rail, ((5+i * 0.2, 0, 0.05), PxIdentity), ((0, 0, 0), PxIdentity), name=f'rootx{i}', damping=0, stiffness=0, type='slider', range=[-np.inf, np.inf], joint_name=f'rootx{i}')
            root3 = self.my_add_link(root2, ((0, 0, 0), x2z), ((0, 0, 0), x2z), name=f'rootz{i}', damping=0, stiffness=0, type='slider', range=[-np.inf, np.inf], joint_name=f'rootz{i}')
            torso = self.my_add_link(root3, ((0, 0, 0), x2y), ((0, 0, 0), x2y), name=f'torso{i}', damping=0, stiffness=0, type='hinge', range=[-np.inf, np.inf], joint_name=f'rooty{i}')
            self.add_sphere(torso, (0, 0, 0), (1, 0, 0, 0), 0.05, np.array((0.8, 0.8, 1)), name=f"xxx{i}")

        wrapper = builder.build(True)
        self.add_force_actuator("rooty0", -3, 3)
        self.sim.add_ground(0)
        return wrapper, None


if __name__ == '__main__':
    #env = Ball()
    ##while True:
    #   env.render()
    pass
