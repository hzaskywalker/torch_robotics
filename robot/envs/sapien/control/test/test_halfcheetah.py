from robot.envs.sapien.control.halfcheetah import HalfCheetahEnv

def test():
    cheetah = HalfCheetahEnv()

    cheetah.reset()
    for i in range(1000):
        action = cheetah.action_space.sample()

        cheetah.step(action)

        cheetah.render()


if __name__ == '__main__':
    test()