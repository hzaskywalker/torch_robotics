# Goal-conditioned Environment...
import gym
import os
from collections import OrderedDict
from sapien.core import Pose
from gym.spaces import Box, Dict
from .simulator import Simulator
from .loader import load_scene
from .arm_wrapper import Arm7DOF
import numpy as np
from .cost import ArmMove



class ArmReach(Simulator):
    def __init__(self, reward_type='dense', eps=0.06, fix_goal=False, timestep=0.0025, frameskip=4):
        Simulator.__init__(self, dt=timestep, frameskip=frameskip)

        assert reward_type in ['dense', 'sparse']
        self.eps = eps

        params = OrderedDict(
            ground=0,
            agent=OrderedDict(
                type='robot',
                lock=['pan_joint', 'tilt_joint', 'linear_joint', 'right_gripper_finger3_joint', 'right_gripper_finger2_joint', 'right_gripper_finger1_joint'],
                lock_value=[0, 0, 0, 0.8, 0.8, 0.8],
                actuator=['right_shoulder_pan_joint',
                          'right_shoulder_lift_joint',
                          'right_arm_half_joint',
                          'right_elbow_joint',
                          'right_wrist_spherical_1_joint',
                          'right_wrist_spherical_2_joint',
                          'right_wrist_3_joint',
                          ],
                actuator_range=[[-50, 50] for i in range(7)],
                ee="right_ee_link",
            ),
        )
        load_scene(self, params)
        self.agent = Arm7DOF(self.objects['agent'], self._actuator_dof['agent'],
                             self._ee_link_idx['agent'], self._actuator_joint['agent'])
        del self._actuator_dof
        del self._ee_link_idx

        if not fix_goal:
            self.goal_space = Box(low=np.array([0.5, -0.4, 0.3]), high=np.array([1., 0.4, 1]))
        else:
            self.goal_space = Box(low=np.array([0.9, 0, 0.8]), high=np.array([0.9+0.0001, 0.0001, 0.8+0.0001]))

        self._goal = self.goal_space.sample()

        self._start_state = self.state_vector().copy()

        obs = self._get_obs()
        self.reward_type = reward_type

        self.observation_space = Dict(
            observation=Box(-np.inf, np.inf, shape=obs['observation'].shape),
            desired_goal=self.goal_space,
            achieved_goal=self.goal_space,
        )

        self.action_range = 50
        self.action_space = Box(-1., 1., self.agent.dofs.shape)
        self.build_goal_vis()
        self.reset()

    def build_goal_vis(self):
        actor_builder = self.scene.create_actor_builder()
        actor_builder.add_box_visual(Pose(), (self.eps, self.eps, self.eps), (0, 0, 1), 'goal')
        box = actor_builder.build(True)
        self.vis_goal = box

    def reset(self):
        self.load_state_vector(self._start_state)
        obs = self._get_obs()
        self._goal = self.goal_space.sample()
        self.vis_goal.set_pose(Pose(self._goal))

        self._reset = True
        self.timestep = 0

        return obs

    def get_jacobian(self):
        return self.agent.compute_jacobian()

    def get_geom(self):
        # removed at 2020/4/14 15:35, see previous commits for better understanding.
        raise NotImplementedError


    def step(self, action):
        # do_simulation
        assert self._reset
        qf = np.array(action).clip(-1, 1) * self.action_range

        for i in range(self.frameskip):
            self.agent.set_qf(qf)
            self.do_simulation()

        obs = self._get_obs()
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'])
        if self.reward_type == 'dense':
            is_success = (-reward < self.eps)
        else:
            is_success = reward > -0.5

        info = {'is_success': is_success}

        return obs, reward, False, info

    def state_vector(self):
        return np.concatenate([super(ArmReach, self).state_vector(), self._goal], axis=0)

    def load_state_vector(self, state):
        self._goal = state[-3:]
        super(ArmReach, self).load_state_vector(state[:-3])

    def _get_obs(self):
        # I don't store the qacc as it's not accurate
        observation = np.concatenate([self.agent.get_qpos(), self.agent.get_qvel()])

        achieved_goal = self.agent.get_ee_links().pose.p
        desired_goal = self._goal.copy()


        obs = {
            'observation': observation,
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal
        }
        return obs


    def compute_reward(self, achieved_goal, desired_goal, info=None):
        # TODO: hack now, I don't want to implement a pickable reward system...
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        if self.reward_type == 'dense':
            return - d
        else:
            return -(d > self.eps).astype(np.float32)


class ArmReachWithXYZ(ArmReach):
    def build_goal_vis(self):
        super(ArmReachWithXYZ, self).build_goal_vis()

        end_ee_builder = self.scene.create_actor_builder()
        end_ee_builder.add_box_visual(Pose(), (self.eps, self.eps, self.eps), (1, 0, 0), 'end_ee')
        box = end_ee_builder.build(True)
        self.end_ee = box

    def _get_obs(self):
        obs = super(ArmReachWithXYZ, self)._get_obs()
        obs['observation'] = np.concatenate((obs['observation'], obs['achieved_goal']))
        return obs

    def render_obs(self, obs, reset=True):
        # state is in fact the observation
        # reset = False to speed up
        if reset:
            _state = self.state_vector()
        state = obs['observation']

        tmp_qpos = self.agent.get_qpos()
        self.agent.set_qpos(state[:len(tmp_qpos)])
        achieved = state[-3:]

        if 'achived_goal' in obs:
            achieved = obs['achieved_goal']
        self.end_ee.set_pose(Pose(achieved))

        if 'desired_goal' in obs:
            goal = obs['desired_goal']
            self.vis_goal.set_pose(Pose(goal))

        img = self.render(mode='rgb_array')
        if reset:
            self.load_state_vector(_state)
        return img

