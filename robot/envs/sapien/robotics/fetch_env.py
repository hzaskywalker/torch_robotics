import numpy as np

from .robot_env import sapien_core, Pose
from .movo_env import MovoEnv
from transforms3d import quaternions


class FetchEnv(MovoEnv):
    def _env_setup(self, initial_qpos):
        super(FetchEnv, self)._env_setup(initial_qpos)
        for idx, i in enumerate(self.model.get_links()):
            if i.name == self.gripper_link.name:
                break
        self.ee_idx = slice(idx * 6, idx*6+6)

        self._actuator_range = np.array(
            [[-1, -1, -1, -1],
            [1, 1, 1, 1]],
        ).T
        #self._actuator_index = []

    def plant_dynamics(self, x, u):
        raise NotImplementedError

    def finite_jacobian(self, q, link=None):
        state = self.get_state() # only get the state of the scene...

        link = self.gripper_link if link is None else link
        q = self.model.get_qpos()
        out = np.zeros((6, q.shape[0]))
        for ii in range(q.shape[0]):
            inc_q = q.copy()
            eps = 1e-4  # finite differences epsilon
            inc_q[ii] += eps
            self.model.set_qpos(inc_q)
            state_inc = np.concatenate((link.pose.p, link.pose.q))

            inc_q[ii] -= 2*eps
            self.model.set_qpos(inc_q)
            state_dec = np.concatenate((link.pose.p, link.pose.q))

            velocity = state_inc[:3] - state_dec[:3]
            angular_velocity_quat = quaternions.qmult(state_inc[3:], quaternions.qinverse(state_dec[3:]))
            vector, theta = quaternions.quat2axangle(angular_velocity_quat)
            assert np.abs(np.linalg.norm(vector) - 1) < 1e-6
            angular_velocity = theta * vector
            out[:, ii] = np.concatenate((velocity, angular_velocity))/ (2*eps)

        self.set_state(state)
        return out

    def _reset_sim(self):
        flag = MovoEnv._reset_sim(self)
        self.integral = 0
        return flag

    def _step_callback(self):
        super(FetchEnv, self)._step_callback()
        self.integral += self.dt * (self.targets - self.model.get_qpos()[self._actuator_index])
        #print(self.integral)

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 1
        rot_ctrl = np.array([1, 0, 0, 0])
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        else:
            raise NotImplementedError

        # TODO: only xyz control now
        jac = self.model.compute_jacobian()[self.ee_idx] # in joint space
        #finite_jac = self.finite_jacobian(self.model.get_qpos())

        jac = jac[:3, self._actuator_index]
        delta = np.linalg.lstsq(jac, pos_ctrl)[0] # in joint space
        targets = self.model.get_qpos()[self._actuator_index] + delta * 10

        joints = self.model.get_joints()
        for target, index in zip(targets, self._actuator_joint_map):
            joints[index].set_drive_property(10000, 10000)
            joints[index].set_drive_target(target)

        qf = self.model.compute_passive_force() # compute the passive force
        qf[self._actuator_index] += self.integral * 250
        self.model.set_qf(qf)

        self.targets = targets


    #def _load_scene(self) -> None:
    #    return None
