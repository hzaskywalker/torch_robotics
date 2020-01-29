# run td3 about the environments...
import numpy as np
import argparse
import gym
from robot.controller.td3 import td3
from robot.envs.sapien.exp.utils import make, RLRecorder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--env_name', type=str, default='cartpole')
    parser.add_argument('--start_timesteps', type=int, default=10000)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--max_timesteps', type=int, default=1000000)
    parser.add_argument('--expl_noise', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--policy_noise', type=float, default=0.2)
    parser.add_argument('--noise_clip', type=float, default=0.5)
    parser.add_argument('--policy_freq', type=int, default=2)

    parser.add_argument('--path', type=str, default='/tmp/tmp')
    args = parser.parse_args()

    env = make(args.env_name)
    print(env.action_space)
    print(env.observation_space)

    # TODO: note we use the same environment for testing... which is troublesome.
    # TODO: if we can run multiple env together, we can must
    recorder = RLRecorder(env, args.path, save_model=slice(0, None, 10), network_loss=slice(0, None, 10),
                          evaluate=slice(0, None, 50), save_video=1, max_timestep=args.max_timesteps)

    td3(env, args.seed, args.start_timesteps, args.eval_freq, args.max_timesteps, args.expl_noise, args.batch_size,
        args.discount, args.tau, args.policy_noise, args.noise_clip, args.policy_freq, recorder=recorder)


if __name__ == '__main__':
    main()