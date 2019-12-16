# code for cartpole
# TODO: wrap cartpole into a control environment
import argparse
from robot.envs import make
from robot.controller.cem import cem

def main():
    env = make('CartPole-v0')
    state = env.reset()
    actions = cem(env, state, 20, iter_num=5, num_mutation=100, num_elite=10)

if __name__ == '__main__':
    main()
