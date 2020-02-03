from robot.envs.sapien.robotics.movo_xyz.movo_reach_xyz_env import MoveReachXYZEnv

def test():
    env = MoveReachXYZEnv('dense')
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
