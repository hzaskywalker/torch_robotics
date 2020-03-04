import argparse
import os
from robot.envs.hyrule.rl_env import RLEnv

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    env = RLEnv('scenes')

    obs = env.reset()
    achieved = obs['achieved_goal']
    desired_goal = obs['desired_goal']
    """
    print(achieved, desired_goal)

    print(env.get_current_cost().cost(env))
    print(env.compute_reward(achieved, desired_goal, None))
    exit(0)
    """


if __name__ == '__main__':
    main()
