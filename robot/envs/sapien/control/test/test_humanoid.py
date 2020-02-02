from robot.envs.sapien.control import HumanoidEnv

def test():
    swimmer = HumanoidEnv()
    print(swimmer.observation_space)
    print(swimmer.action_space)

    swimmer.reset()
    for i in range(10000):
        action = swimmer.action_space.sample()
        action *= 0
        swimmer.step(action)
        swimmer.render()


if __name__ == '__main__':
    test()
