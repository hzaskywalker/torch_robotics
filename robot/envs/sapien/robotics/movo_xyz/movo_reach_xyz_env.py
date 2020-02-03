from ..fetch_env import FetchEnv
import numpy as np
from gym import utils


class MoveReachXYZEnv(FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = np.array([0., -1.381, 0, 0.05, -0.9512, 0.387, 0.608, 2.486, 0.986, 0.986, 0.986, 0., 0.])
        FetchEnv.__init__(
            self, "all_robot", has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
