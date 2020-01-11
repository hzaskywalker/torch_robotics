from robot.envs.sapien.control.ant import AntEnv

def test():
    ant = AntEnv()
    ant.reset()

    for i in range(1000):
        action = ant.action_space.sample()

        ant.step(action)

        ant.render()


if __name__ == '__main__':
    test()