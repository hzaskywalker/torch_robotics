from robot.envs.sapien.control.cartpole import CartpoleEnv

def test():
    cartpole = CartpoleEnv()
    print(cartpole.observation_space)
    print(cartpole.action_space)

    cartpole.reset()
    for i in range(500):
        action = cartpole.action_space.sample()
        cartpole.step(action)
        cartpole.render()


if __name__ == '__main__':
    test()
