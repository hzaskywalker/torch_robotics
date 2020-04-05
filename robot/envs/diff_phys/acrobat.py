# make
import torch
import gym
from gym import utils
from gym.spaces import Dict, Box
from .engine import Articulation2D
from .cv2_rendering import cv2Viewer
import numpy as np
from robot import torch_robotics as tr

class GoalAcrobat(gym.Env, utils.EzPickle):
    def __init__(self, reward_type='dense', eps=0.1, batch_size=1, n_links=2):
        gym.Env.__init__(self)
        utils.EzPickle.__init__(self)

        self.eps = eps
        self.n_links = n_links
        self.total_length = n_links * 1
        self.reward_type = reward_type

        goal_space = Box(low=np.array([-self.total_length, -self.total_length]),
                         high=np.array([self.total_length, self.total_length]))

        self.observation_space = Dict({
            'observation': Box(low=-np.inf, high=np.inf, shape=(n_links * 2 + 2,)),
            'desired_goal': goal_space,
            'achieved_goal': goal_space
        })
        self.action_space = Box(low=-1, high=1, shape=(n_links,))
        self.action_range = 50 #200
        self.velocity_range = 20 #200
        self.batch_size = batch_size


        self.build_model()
        self.dt = self.articulator.timestep
        self.init_qpos = self.articulator.get_qpos().detach().cpu().numpy()
        self.init_qvel = self.articulator.get_qvel().detach().cpu().numpy()

        self.viewer = None
        self._viewers = {}

        self.reset()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                from .rendering import Viewer
                #self.viewer = mujoco_py.MjViewer(self.sim)
                self.viewer = Viewer(500, 500)
            elif mode == 'rgb_array':
                self.viewer = cv2Viewer(500, 500)

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer


    def viewer_setup(self):
        bound = 2.2  # 2.2 for default
        self.viewer.set_bounds(-bound, bound, -bound, bound)


    def render(self, mode='human'):
        viewer = self._get_viewer(mode)
        viewer.draw_line((-2.2, 1), (2.2, 1))
        self.articulator.draw_objects(viewer)
        circle = self.viewer.draw_circle(0.1)

        g = self._goal
        if len(g.shape) > 1:
            g = g[0]
        circle.add_attr(self.viewer.Transform(0, g))
        return viewer.render(return_rgb_array = mode=='rgb_array')

    def render_obs(self, obs, reset=False):
        state = self.state_vector()

        self.set_state(obs['observation'][...,:2], obs['observation'][...,2:4])
        if 'desired_goal' in obs:
            self._goal = obs['desired_goal']
        img = self.render(mode='rgb_array')
        #TODO: render achieved_goal
        self.load_state_vector(state)
        return img


    def build_model(self):
        articulator = Articulation2D(timestep=0.1, max_velocity=self.velocity_range)

        M01 = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, -0.5],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        # v = -w x q
        # we choose q = [0, -0.5, 0] =>
        w, q = [0, 0, 1], [0, 0.5, 0]
        screw1 = w + (-np.cross(w, q)).tolist()
        # m = 1
        G1 = np.diag([1, 1, 1, 1, 1, 1])
        link1 = articulator.add_link(M01, screw1)
        link1.set_inertial(np.array(G1))
        link1.add_box_visual([0, 0, 0], [0.1, 0.5, 0], (0, 0.8, 0.8))
        link1.add_circle_visual((0, 0.5, 0), 0.1, (0.8, 0.8, 0.))

        if self.n_links > 1:
            M12 = np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, -1.0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
            screw2 = screw1
            # m = 1
            G2 = np.diag([1, 1, 1, 1, 1, 1])
            link2 = articulator.add_link(M12, screw2)
            link2.set_inertial(np.array(G2))
            link2.add_box_visual([0, 0, 0], [0.1, 0.5, 0], (0, 0.8, 0.8))
            link2.add_circle_visual((0, 0.5, 0), 0.1, (0.8, 0.8, 0.))

        EE = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, -0.5],
                [0, 0, 1, 0.],
                [0, 0, 0, 1.],
            ]
        )
        articulator.set_ee(EE)
        articulator.build()

        self.articulator = articulator
        self._ee_cache = None


    def set_state(self, qpos, qvel):
        self.articulator.set_qpos(qpos)
        self.articulator.set_qvel(qvel)


    def reset(self):
        if self.batch_size == 1:
            shape = np.shape(self.init_qpos)
            qpos = self.init_qpos + np.random.uniform(low=-0.01, high=0.01, size=shape)
            qvel = self.init_qvel + np.random.uniform(low=-0.01, high=0.01, size=shape)
        else:
            shape = (self.batch_size,) + np.shape(self.init_qpos)
            qpos = self.init_qpos[None,:] + np.random.uniform(low=-0.01, high=0.01, size=shape)
            qvel = self.init_qpos[None,:] + np.random.uniform(low=-0.01, high=0.01, size=shape)
        self.set_state(qpos, qvel)

        self._timestep = 0

        r = (np.random.random(self.batch_size) * 0.6+0.4) * self.total_length
        theta = np.random.random(self.batch_size) * np.pi * 2 - np.pi
        self._goal = np.stack((np.sin(theta) * r, np.cos(theta) * r), axis=1)
        if self.batch_size == 1:
            self._goal = self._goal[0]
        return self._get_obs()

    def get_ee(self):
        # could be accelerated
        Ts = self.articulator.forward_kinematics()
        return Ts[..., -1, :2, 3]

    def _get_obs(self):
        q = self.articulator.get_qpos().detach().cpu().numpy()
        qvel = self.articulator.get_qvel().detach().cpu().numpy()
        achieved_goal = self.get_ee().detach().cpu().numpy()

        return {
            'observation': np.concatenate([q, qvel, achieved_goal], axis=-1),
            'desired_goal': np.array(self._goal).copy(),
            'achieved_goal': achieved_goal.copy(),
        }

    def step(self, action):
        action = action.clip(-1, 1)
        self.articulator.set_qf(action * self.action_range)
        with torch.no_grad():
            self.articulator.step()

        ob = self._get_obs()
        reward = self.compute_reward(ob['achieved_goal'], ob['desired_goal'])
        if self.reward_type == 'dense':
            is_success = (-reward < self.eps)
        else:
            is_success = reward > -0.5

        done = False
        return ob, reward, done, {'is_success': is_success}


    def compute_reward(self, achieved_goal, desired_goal, info=None):
        # TODO: hack now, I don't want to implement a pickable reward system...
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        if self.reward_type == 'dense':
            return -d
        else:
            return -(d > self.eps).astype(np.float64)

    def get_jacobian(self):
        # only return the jacobian for end effector position
        jac = self.articulator.compute_jacobian()
        if len(jac.shape) == 2:
            jac = jac[None,:]
        #return self.articulator.compute_jacobian().detach().cpu().numpy()
        achieved = self.get_ee().detach().cpu().numpy()
        #xx = np.array((achieved[0], achieved[1], 0))
        xx = np.concatenate((achieved, achieved[..., -1:]*0), axis=-1)
        q_ee = torch.tensor(xx, dtype=torch.float64, device='cuda:0')
        if len(q_ee.shape) == 1:
            q_ee = q_ee[None,:]

        jac = -tr.dot(tr.vec_to_so3(q_ee), jac[:,:3]) + jac[:,3:]
        return jac.detach().cpu().numpy()

    def compute_inverse_dynamics(self, qacc):
        if isinstance(qacc, np.ndarray):
            qacc = torch.tensor(qacc, device=self.articulator.qpos.device, dtype=torch.float64)
        return self.articulator.inverse_dynamics(self.articulator.qpos, self.articulator.qvel, qacc).detach().cpu().numpy()

    def state_vector(self):
        qpos = self.articulator.get_qpos().detach().cpu().numpy()
        qvel = self.articulator.get_qvel().detach().cpu().numpy()
        return np.concatenate((qpos, qvel, self._goal), axis=-1)

    def load_state_vector(self, state):
        qpos = state[..., :self.init_qpos.shape[-1]]
        qvel = state[..., self.init_qpos.shape[-1]:self.init_qpos.shape[-1] + self.init_qvel.shape[-1]]
        self._goal = state[..., -self._goal.shape[-1]:]
        self.set_state(qpos, qvel)


class IKController:
    def __init__(self, env):
        self.env = env.unwrapped

    def __call__(self, state):
        state_vector = self.env.state_vector()
        state, goal = state['observation'], state['desired_goal']

        dim = (state.shape[-1] - 2)//2
        qpos = state[..., :dim]
        qvel = state[..., dim:dim*2]
        achieved = state[..., dim*2:]

        self.env.set_state(qpos, qvel)

        jac = self.env.get_jacobian()  # notice that the whole system is (w, v)

        def togpu(x): return torch.tensor(x, device='cuda:0', dtype=torch.float64)
        cartesian_diff = togpu((goal-achieved)[..., :2])
        is_single = 0
        if len(cartesian_diff.shape) == 1:
            is_single = 1
            cartesian_diff = cartesian_diff[None,:]
        q_delta = tr.dot(torch.pinverse(togpu(jac[:, :2])), cartesian_diff)

        qvel = togpu(qvel)
        qacc = (q_delta - qvel)/self.env.dt

        if is_single:
            qacc = qacc[0]
        qf = self.env.compute_inverse_dynamics(qacc)

        self.env.load_state_vector(state_vector)
        action = qf/self.env.action_range

        return action
