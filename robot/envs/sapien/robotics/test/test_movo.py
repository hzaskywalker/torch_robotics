import argparse
import numpy as np
from robot.envs.sapien.robotics.movo_env import MovoEnv

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onscreen', default=1, type=int)
    args = parser.parse_args()

    init_qpos = np.array([0., -1.381, 0, 0.05, -0.9512, 0.387, 0.608, 2.486, 0.986, 0.986, 0.986, 0., 0.])
    env = MovoEnv("all_robot", n_substeps=1, initial_qpos=init_qpos, block_gripper=True, has_object=True)

    print(env.action_space)
    print(env.observation_space)
    obs = env.reset()
    print([obs[i].shape for i in obs])

    for i in range(10000):
        if not args.onscreen:
            img = env.render(mode='rgb_array')
            # print(img.shape, img.dtype, img.min(), img.max())
            # yield img
            import cv2
            cv2.imshow('x', img)
            cv2.waitKey(2)
        else:
            img = env.render()

        a = env.action_space.sample()
        t, r, d, _ = env.step(a)


if __name__ == '__main__':
    test()