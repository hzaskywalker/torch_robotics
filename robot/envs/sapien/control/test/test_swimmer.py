from robot.envs.sapien.control.swimmer import SwimmerEnv

def test():
    swimmer = SwimmerEnv()
    print(swimmer.observation_space)
    print(swimmer.action_space)

    swimmer.reset()
    for i in range(10000):
        action = swimmer.action_space.sample()
        swimmer.step(action)
        swimmer.render()


if __name__ == '__main__':
    test()
