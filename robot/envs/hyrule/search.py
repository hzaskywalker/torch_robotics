import torch
import cv2
import argparse
import numpy as np
import tqdm
from robot.controller.rollout_controller import RolloutCEM
from robot.utils.data_parallel import DataParallel
from collections import OrderedDict

from robot.envs.hyrule import make, dump_json, load_json

class Rollout:
    def __init__(self, make, env_name):
        self.env = make(env_name)

    def __call__(self, s, a):
        rewards = []
        for s, a in zip(s, a):
            self.env.unwrapped.load_state_vector(s)
            # I am not sure if this is correct
            self.env._elapsed_steps = int(s[0])
            reward = 0
            for action in a:
                s, r, done, _ = self.env.step(action)
                reward -= r
                if done:
                    break
            rewards.append(reward)
        return np.array(rewards)

class SapienMujocoRolloutModel:
    def __init__(self, env_name, n=20):
        self.model = DataParallel(n, Rollout, make, env_name)

    def rollout(self, s, a):
        is_cuda =isinstance(a, torch.Tensor)
        if is_cuda:
            device = s.device
            s = s.detach().cpu().numpy()
            a = a.detach().cpu().numpy()
        r = self.model(s, a)
        if is_cuda:
            r = torch.tensor(r, dtype=torch.float, device=device)
        return None, r

def eval_policy(policy, eval_env, eval_episodes=10, save_video=0, video_path="video{}.avi", timestep=int(1e9)):

    avg_reward = 0.
    acc = []

    trajectories = []
    for episode_id in tqdm.trange(eval_episodes):
        state, done = eval_env.reset(), False

        out = None
        if isinstance(policy, object):
            if 'reset' in policy.__dir__():
                policy.reset()

        #while not done:
        states = []
        actions = []
        for i in tqdm.trange(timestep):
            if episode_id < save_video:
                if video_path[-3:] == 'avi':
                    img = eval_env.render(mode='rgb_array')
                    if out is None:
                        out = cv2.VideoWriter(
                            video_path.format(episode_id), cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (img.shape[1], img.shape[0]))
                    out.write(img)
                else:
                    eval_env.render()

            state = np.array(eval_env.unwrapped.state_vector())
            states.append(state.tolist())
            action = policy(state)
            actions.append(action.tolist())
            state, reward, done, info = eval_env.step(action)
            avg_reward += reward
            if done:
                break
        states.append(state.tolist())

        if out is not None:
            out.release()
        trajectories.append([states, actions])

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    if len(acc) > 0:
        print(f"Evaluation success rate over {eval_episodes} episodes: {np.mean(acc):.3f}")
    print("---------------------------------------")
    return trajectories


def add_parser(parser):
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--env_name', type=str, default='halfcheetah')
    parser.add_argument('--iter_num', type=int, default=5)
    parser.add_argument('--initial_iter', type=int, default=0)
    parser.add_argument('--num_mutation', type=int, default=400)
    parser.add_argument('--num_elite', type=int, default=20)
    parser.add_argument('--std', type=float, default=0.5)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--num_proc', type=int, default=20)
    parser.add_argument('--video_num', type=int, default=1)
    parser.add_argument('--video_path', type=str, default='video{}.avi')
    parser.add_argument('--output_path', type=str, default='tmp.json')
    parser.add_argument('--num_test', type=int, default=1)
    parser.add_argument('--timestep', type=int, default=100)
    parser.add_argument('--add_actions', type=int, default=1)
    parser.add_argument('--controller', type=str, default='cem', choices=['cem', 'poplin'])


def main():
    #pass
    parser = argparse.ArgumentParser()
    add_parser(parser)
    args = parser.parse_args()

    model = SapienMujocoRolloutModel(args.env_name, n=args.num_proc)

    env = make(args.env_name)
    controller = RolloutCEM(model, action_space=env.action_space,
                            add_actions=args.add_actions,
                            horizon=args.horizon, std=args.std,
                            iter_num=args.iter_num,
                            initial_iter=args.initial_iter,
                            num_mutation=args.num_mutation, num_elite=args.num_elite, alpha=0.1, trunc_norm=True, lower_bound=env.action_space.low, upper_bound=env.action_space.high)
    trajectories = eval_policy(controller, env, args.num_test, args.video_num, args.video_path, timestep=args.timestep)

    params = load_json(args.env_name)
    params['trajectories'] = trajectories
    dump_json(args.output_path, params)


if __name__ == '__main__':
    main()
