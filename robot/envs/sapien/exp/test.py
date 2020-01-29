from robot.envs.sapien.exp.utils import eval_policy, make
import torch
import argparse

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--eval_episode', type=int, default=10)
    parser.add_argument('--video_path', type=str, default='video{}.mp4')
    parser.add_argument('--save_video', type=int, default=0)
    args = parser.parse_args()

    policy = torch.load(args.path)

    eval_policy(policy, args.env_name,
                args.seed, args.eval_episode,
                args.save_video,
                args.video_path)

if __name__ == '__main__':
    test()