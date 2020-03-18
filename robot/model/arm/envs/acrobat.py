from robot.envs.sapien.control import sapien_env
from robot.envs.sapien.control.sapien_env import sapien_core, Pose
from gym import utils
from gym.spaces import Dict, Box
import numpy as np

class GoalAcrobat(sapien_env.SapienEnv, utils.EzPickle):
    def __init__(self, eps=0.05,
                 length=np.array([0.5, 0.5]),
                 reward_type='dense'):

        self._actuator_dof = {'agent': np.arange(len(length))}
        self.actuator_range_val = 50
        self._actuator_range = {'agent': np.array([[-self.actuator_range_val, self.actuator_range_val]
                                                   for i in range(len(length))])}

        self.eps = eps
        self.reward_type = reward_type

        self.total_length = np.sum(length)

        goal_space = Box(low=np.array([-self.total_length, -self.total_length]),
                         high=np.array([self.total_length, self.total_length]))
        self._goal = goal_space.sample()
        self._timestep = 0

        self._length = length

        sapien_env.SapienEnv.__init__(self, 4, timestep=0.025)
        utils.EzPickle.__init__(self)

        # The goal is to stablize at one location.
        self.observation_space = Dict({
            'observation': Box(low=-np.inf, high=np.inf, shape=(6,)),
            'desired_goal': goal_space,
            'achieved_goal': goal_space
        })

    def _get_obs(self):
        #self.ee.set_pose()
        q = self.model.get_qpos().flat
        qvel =self.model.get_qvel().flat
        #print(np.array(q))

        self.goal_sphere.set_pose(Pose((self._goal[0], 0, self._goal[1])))

        achieved_goal = self.ee_link.pose.p
        x = achieved_goal[0]
        z= achieved_goal[2]
        self.ee_sphere.set_pose(Pose((x, 0, z)))

        return {
            'observation': np.concatenate([q, qvel, achieved_goal]),
            'desired_goal': np.array(self._goal).copy(),
            'achieved_goal': np.array([achieved_goal[0], achieved_goal[2]]),
        }

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-0.01, high=0.01, size=np.shape(self.init_qpos))
        qvel = self.init_qvel + self.np_random.uniform(low=-0.01, high=0.01, size=np.shape(self.init_qvel))
        self.set_state(qpos, qvel)

        self._timestep = 0
        #self._goal = self.observation_space['desired_goal'].sample()

        r = (np.random.random() * 0.6+0.4) * self.total_length
        theta = np.random.random() * np.pi * 2 - np.pi
        self._goal = np.array((np.sin(theta) * r, np.cos(theta) * r))
        return self._get_obs()

    def step(self, a):
        a = a.clip(-1, 1)
        #print('wa start step')
        self.do_simulation(a * self.actuator_range_val, self.frame_skip)
        #print('finish simulate')
        ob = self._get_obs()
        #print('get_obs start step')

        reward = self.compute_reward(ob['achieved_goal'], ob['desired_goal'])
        if self.reward_type == 'dense':
            is_success = (-reward < self.eps)
        else:
            is_success = reward > -0.5

        done = False
        #print('finish')
        return ob, reward, done, {'is_success': is_success}


    def compute_reward(self, achieved_goal, desired_goal, info=None):
        # TODO: hack now, I don't want to implement a pickable reward system...
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        if self.reward_type == 'dense':
            return -d
        else:
            return -(d > self.eps).astype(np.float32)


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
        x2y = np.array([0.7071068, 0, 0, -0.7071068])

        base = self.add_link(None, Pose(np.array([0, 0, 0]), PxIdentity), "rail")
        tmp = base
        for idx, l in enumerate(self._length):
            tmp = self.my_add_link(tmp, ((0, 0, 0), x2y), ((0, 0, l), x2y),
                                   f'link{idx}', f'joint{idx}', range=[-np.inf, np.inf],
                                   damping=0.5, father_pose_type='sapien')
            self.fromto(tmp, f"0 0 0 0 0 {l}", size=0.03, rgb=np.array([0., 0.7, 0.7]), name='cpole')

        wrapper = builder.build(True)
        for i in range(len(self._length)):
            self.add_force_actuator(f"joint{i}", -1, 1)

        self.goal_sphere = self.build_sphere(self.eps, (1., 0, 0.))
        self.ee_sphere = self.build_sphere(self.eps, (0., 0, 1.))

        self.ee_link = wrapper.get_links()[-1]
        self.agent = wrapper

        return wrapper, None

    def build_sphere(self, eps, color):
        actor_builder = self.sim.create_actor_builder()
        actor_builder.add_sphere_visual(Pose(), eps, color, 'goal')
        box = actor_builder.build(True)
        return box

    def state_vector(self):
        return np.array(
            [self._timestep] + self.model.get_qpos().tolist() + self.model.get_qvel().tolist() +
            list(self._goal))

    def load_state_vector(self, vec):
        self._timestep = int(vec[0])
        d = (vec.shape[0] - 3)//2
        self.model.set_qpos(vec[1:1+d])
        self.model.set_qvel(vec[d+1:1+2*d])
        self._goal = vec[-2:].copy()

    def get_jacobian(self):
        jac = self.model.compute_jacobian()[-6:] # in joint space
        return jac, None


    def render_obs(self, state, reset=True):
        # state is in fact the observation
        # reset = False to speed up
        if reset:
            _state = self.state_vector()
        state = state['observation']

        tmp_qpos = self.agent.get_qpos()
        self.agent.set_qpos(state[:len(tmp_qpos)])
        self.ee_sphere.set_pose(Pose(state[-3:]))

        img = self.render(mode='rgb_array')
        if reset:
            self.load_state_vector(_state)
        return img

