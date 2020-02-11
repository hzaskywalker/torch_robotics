from robot.envs.sapien.robotics.movo_xyz.movo_reach_xyz_env import MoveReachXYZEnv
import numpy as np

def test():
    env = MoveReachXYZEnv('dense')
    print(env.observation_space)
    print(env.action_space)

    env.reset()
    for i in range(10):
        obs = env.reset()
        for i in range(100):
            action = env.action_space.sample()
            action = list(obs['desired_goal'] - obs['achieved_goal']) + [0]
            obs, r, d, _ = env.step(np.array(action))
            print(r)
            env.render()

if __name__ == '__main__':
    test()
