from robot.envs.sapien.robotics.movo import movo_reach

def test():
    env = movo_reach.MoveReachEnv('dense')
    print(env.observation_space)
    print(env.action_space)

    env.reset()
    for i in range(10):
        env.reset()
        for i in range(100):
            action = env.action_space.sample()
            t, r, d, _ = env.step(action)
            print(r)
            env.render()

if __name__ == '__main__':
    test()
