import argparse
import os
from robot.envs.hyrule.rl_env import ArmReach

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    env = ArmReach('dense', jacobian=True)

    obs = env.reset()
    achieved = obs['achieved_goal']
    desired_goal = obs['desired_goal']
    while True:
        env.render()
        action = env.action_space.sample()
        _, reward, _, _ = env.step(action)
        print(reward)


if __name__ == '__main__':
    main()
