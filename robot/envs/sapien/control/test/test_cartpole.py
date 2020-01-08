from robot.envs.sapien.control.cartpole import CartpoleEnv

def test():
    cartpole = CartpoleEnv()

    cartpole.reset()
    for i in range(500):
        action = cartpole.action_space.sample()
        action *= 0

        cartpole.step(action)
        cartpole.render()


if __name__ == '__main__':
    test()
