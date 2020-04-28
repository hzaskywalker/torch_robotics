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

    def step(self, a=None):
        #self.do_simulation(a * 108, self.frame_skip)
        self.sim.step()
        return np.array([1]), np.array([1]), 0, None


    def build_render(self):
        self.sim.set_ambient_light([.4, .4, .4])
        self.sim.set_shadow_light([1, -1, -1], [.5, .5, .5])
        self.sim.add_point_light([2, 2, 2], [1, 1, 1])
        self.sim.add_point_light([2, -2, 2], [1, 1, 1])
        self.sim.add_point_light([-2, 0, 2], [1, 1, 1])

        self._renderer.set_camera_position(5, -3, 2)
        self._renderer.set_camera_rotation(1.57, -0.5)

    def build_model(self):
        PxIdentity = np.array([1, 0, 0, 0])  # rotation
        material = self._sim.create_physical_material(0, 0, 1.0)

        balls = []
        for i in range(2):
            actor = self.sim.create_actor_builder()
            actor.add_sphere_shape(radius=0.05, material=material, density=10000)
            actor.add_sphere_visual(radius=0.05, color=np.array((0.8, 0.8, 1)))
            balls.append(actor.build())

            balls[-1].set_pose(Pose((5+i*0.1, 0, 0.05)))
            balls[-1].set_velocity(np.array([0,0,0]))
            balls[-1].set_angular_velocity(np.array([0,0,0]))

        self.sim.add_ground(0, material=material)
        balls[0].set_pose(Pose((5-2, 0, 0.05)))
        balls[0].set_velocity(np.array([1, 0, 0]))
        #wrapper.set_qpos([-1, 0, 0,0])
        #wrapper.set_qvel([1, 0, 0,0])
        class Fake:
            def get_joints(self):
                return np.array([])
            def get_qf(self):
                return np.array([])
            def get_qpos(self):
                return np.array([])
            def get_qvel(self):
                return np.array([])

        return Fake(), None


if __name__ == '__main__':
    env = Ball()
    while True:
        env.step()
        env.render()
