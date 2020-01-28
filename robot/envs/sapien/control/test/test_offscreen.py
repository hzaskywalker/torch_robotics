from robot.envs.sapien.control.halfcheetah import HalfCheetahEnv
import tqdm

def test_cheetah():
    cheetah = HalfCheetahEnv()

    print(cheetah.observation_space)
    print(cheetah.action_space)

    cheetah.reset()
    for i in tqdm.trange(1000):
        action = cheetah.action_space.sample()

        cheetah.step(action)

        img = cheetah.render(mode='rgb_array')
        import cv2
        cv2.imshow('x', img)
        cv2.waitKey(1)


if __name__ == '__main__':
    test_cheetah()