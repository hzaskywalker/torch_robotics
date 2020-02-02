import argparse
from robot.envs.sapien.control import SwimmerEnv
import tqdm


def test_cheetah():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onscreen', default=0, type=int)
    args = parser.parse_args()

    #cheetah = HalfCheetahEnv()
    #cheetah = AntEnv()
    cheetah = SwimmerEnv()

    print(cheetah.observation_space)
    print(cheetah.action_space)

    cheetah.reset()
    def work():
        for i in tqdm.trange(1000):
            action = cheetah.action_space.sample()

            cheetah.step(action)

            if not args.onscreen:
                img = cheetah.render(mode='rgb_array')
                #print(img.shape, img.dtype, img.min(), img.max())
                #yield img
                print(img.min(), img.max())
                import cv2
                cv2.imshow('x', img)
                cv2.waitKey(2)
            else:
                img = cheetah.render()
    work()
    #write_video(work(), path='tmp.avi')


if __name__ == '__main__':
    test_cheetah()