from gym.envs.mujoco import mujoco_env
from gym import utils
import numpy as np
from mujoco_py import MjViewer, MjRenderContextOffscreen

class HumanoidStandupEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.env_name = 'humanoid_standup'
        self.seeding = False
        self.real_step = True
        self.env_timestep = 0
        self.mujoco_render_frames = False
        self.observation_dim = 376
        self.action_dim = 17
        mujoco_env.MujocoEnv.__init__(self, 'humanoidstandup.xml', 5)
        utils.EzPickle.__init__(self)
        self.video = []

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        pos_after = self.sim.data.qpos[2]
        data = self.sim.data
        uph_cost = (pos_after - 0) / self.model.opt.timestep

        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = uph_cost - quad_ctrl_cost - quad_impact_cost + 1

        done = bool(False)
        if self.mujoco_render_frames is True:
            camera_id = self.model.camera_name2id('track')
            try:
                #self.viewer._record_video = True
                #self.viewer.render()
                img = self.viewer.render(640, 480, 0, camera_id)
                data = self.viewer.read_pixels(640, 480, depth=False)
                img = data[::-1]
                import cv2
                cv2.imshow('x', img)
                cv2.waitKey(1)
            except:
                self.viewer_setup()
                #self.viewer._record_video = True
                #self.viewer.__video_path = "/home/tonyyang/Desktop/video_0.mp4"
                self.viewer._run_speed = 0.5
                #self.viewer._run_speed /= self.frame_skip
                #self.viewer.render()
                import cv2
                img = self.viewer.render(640, 480, camera_id)
                data = self.viewer.read_pixels(640, 480, depth=False)
                img = data[::-1]
                cv2.imshow('x', img)
                cv2.waitKey(1)

        return self._get_obs(), reward, done, dict(reward_linup=uph_cost, reward_quadctrl=-quad_ctrl_cost, reward_impact=-quad_impact_cost)

    def reset_model(self, seed = None):
        if seed is not None:
            self.seeding = True
            self.seed(seed)
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer = MjRenderContextOffscreen(self.sim, -1)
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 0.8925
        self.viewer.cam.elevation = -20

    def get_env_state(self):
        return dict(qp=self.data.qpos.copy(), qv=self.data.qvel.copy(), timestep=self.env_timestep)

    def set_env_state(self, state):
        self.sim.reset()
        qp = state['qp'].copy()
        qv = state['qv'].copy()
        self.set_state(qp, qv)
        self.env_timestep = state['timestep']
        self.sim.forward()

    def get_env_infos(self):
        return dict(state=self.get_env_state())


