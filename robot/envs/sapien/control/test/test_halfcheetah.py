from robot.envs.sapien.control.halfcheetah import HalfCheetahEnv

def test_cheetah():
    cheetah = HalfCheetahEnv()

    print(cheetah.observation_space)
    print(cheetah.action_space)

    cheetah.reset()
    for i in range(1000):
        action = cheetah.action_space.sample()

        cheetah.step(action)

        cheetah.render()


if __name__ == '__main__':
    test_cheetah()