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
        self._actuator_index = []

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]
        print('action', action)

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
        jac = self.model.compute_jacobian()[self.ee_idx]
        print(jac.sum(axis=0))
        raise NotImplementedError
        qf = np.linalg.lstsq(jac, pos_ctrl)[0]
        self.model.set_qf(qf)
