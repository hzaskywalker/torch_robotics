import argparse
import os
from robot.envs.hyrule.rl_env import ArmReach

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    env = ArmReach('sparse', jacobian=False)

    obs = env.reset()
    achieved = obs['achieved_goal']
    desired_goal = obs['desired_goal']
    idx = 0
    while True:
        idx += 1
        if idx % 50 == 1:
            env.reset()
        env.render()
        action = env.action_space.sample()
        _, reward, _, info = env.step(action)
        print(reward, info['is_success'])


if __name__ == '__main__':
    main()
