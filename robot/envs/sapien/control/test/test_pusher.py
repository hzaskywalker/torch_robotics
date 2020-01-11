from robot.envs.sapien.control.pusher import PusherEnv

def test():
    pusher = PusherEnv()
    print(pusher.observation_space)
    print(pusher.action_space)

    pusher.reset()
    for i in range(10000):
        action = pusher.action_space.sample()
        action *= 0
        pusher.step(action)
        pusher.render()


if __name__ == '__main__':
    test()
