import numpy as np

from .robot_env import sapien_core, Pose
from .movo_env import MovoEnv

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

    def finite_differences(self, x, u):
        """ calculate gradient of plant dynamics using finite differences
        x np.array: the state of the system
        u np.array: the control signal
        """
        dof = u.shape[0]
        num_states = x.shape[0]

        A = np.zeros((num_states, num_states))
        B = np.zeros((num_states, dof))

        eps = 1e-4  # finite differences epsilon
        for ii in range(num_states):
            # calculate partial differential w.r.t. x
            inc_x = x.copy()
            inc_x[ii] += eps
            state_inc = self.plant_dynamics(inc_x, u.copy())
            dec_x = x.copy()
            dec_x[ii] -= eps
            state_dec = self.plant_dynamics(dec_x, u.copy())
            A[:, ii] = (state_inc - state_dec) / (2 * eps)

        for ii in range(dof):
            # calculate partial differential w.r.t. u
            inc_u = u.copy()
            inc_u[ii] += eps
            state_inc = self.plant_dynamics(x.copy(), inc_u)
            dec_u = u.copy()
            dec_u[ii] -= eps
            state_dec = self.plant_dynamics(x.copy(), dec_u)
            B[:, ii] = (state_inc - state_dec) / (2 * eps)

        return A, B

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]
        print('action', action)

        #pos_ctrl /= self.dt
        pos_ctrl *= 1
        #pos_ctrl *= 0.1  # limit maximum change in position
        #rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        rot_ctrl = np.array([1, 0, 0, 0])
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        else:
            raise NotImplementedError
        #action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # TODO: only xyz control now
        #print(self.model.compute_jacobian())
        #print(self._actuator_index)
        jac = self.model.compute_jacobian()[self.ee_idx][:3, self._actuator_index] # in joint space
        delta = np.linalg.lstsq(jac, pos_ctrl)[0] # in joint space
        targets = self.model.get_qpos()[self._actuator_index] - delta # target qpos

        joints = self.model.get_joints()
        for target, index in zip(targets, self._actuator_joint_map):
            joints[index].set_drive_property(1000000, 1000000)
            joints[index].set_drive_target(target)
        #super(FetchEnv, self)._set_action(real_action)
        #joints = self.model.get_links().get_joint()


    def _load_scene(self) -> None:
        return None
