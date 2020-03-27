# only predict the next q
import os
import argparse
import tqdm
import numpy as np
from robot.controller.rollout_controller import RolloutCEM

import torch
from robot.model.arm.dataset import Dataset
from robot.model.arm.envs import make
from robot.model.arm.frame import make_frame_cls
from robot.model.arm.models import make_model
from robot.model.arm.agents import make_agent
from robot import U


class DatasetWrapper:
    def __init__(self, dataset, batch_size, timestep, cls):
        self.dataset = dataset
        self.batch_size = batch_size
        self.timestep = timestep
        self.cls = cls

    def sample(self, mode):
        obs, action = self.dataset.sample(mode=mode, batch_size=self.batch_size, timestep=self.timestep, use_geom=False)
        frames = self.cls.from_observation(obs)
        return frames[..., 0, :], action, frames[..., 1:, :]


class RolloutWrapper:
    def __init__(self, model, frame_type):
        self.model = model
        self.cls = frame_type

    def rollout(self, obs, a, goal):
        obs = U.togpu(obs)
        a = U.togpu(a)
        if goal is not None:
            goal = U.togpu(goal)

        s = self.cls.from_observation(obs)
        predict, reward = self.model.rollout(s, a, goal)
        return predict, -reward


class Renderer:
    # renderer is not the state
    def __init__(self, env):
        self.env = env

    def render_frame(self, x):
        obs = x.as_observation()
        if self.goal is not None:
            obs['desired_goal'] = self.goal
        return self.env.unwrapped.render_obs(obs, reset=False)

    def render(self, data, info, num=2, return_image=True, goal=None):
        self.goal = goal
        start, actions, future = [i[:num] for i in data]

        start = start.cpu()
        future = future.cpu()

        predict = info['predict']

        images = []
        for i in range(num):
            start_img = self.render_frame(start[i])
            ground_truth = [start_img] + [self.render_frame(s) for s in future[i]]
            predicted = [start_img]+ [self.render_frame(s) for s in predict[i]]
            if return_image:
                images.append(np.concatenate((
                    np.concatenate(ground_truth, axis=1),
                    np.concatenate(predicted, axis=1),
                ), axis=0))
            else:
                def make():
                    for a, b in zip(ground_truth, predicted):
                        yield np.concatenate((a, b), axis=1)
                return make()
        return np.stack(images)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default='arm', choices=['arm', 'acrobat2', 'plane'])
    parser.add_argument("--batchnorm", type=int, default=0)
    parser.add_argument("--model", type=str, default='mlp')
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--num_epoch", type=int, default=1000)
    parser.add_argument("--num_train_iter", type=int, default=10000)
    parser.add_argument("--num_valid_iter", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--path", type=str, default='rollout')
    parser.add_argument("--timestep", type=int, default=10)
    parser.add_argument("--device", type=str, default='cuda:0')

    parser.add_argument("--weight_q", type=float, default=1.)
    parser.add_argument("--weight_dq", type=float, default=1.)
    parser.add_argument("--weight_ee", type=float, default=1.)
    parser.add_argument("--resume", type=int, default=0)
    args = parser.parse_args()
    return args


class trainer:
    def __init__(self, args):
        self.args = args
        self.path = self.args.path
        self.env = make(args.env_name)
        dataset_path = os.path.join('/dataset/', args.env_name)
        self.frame_type = make_frame_cls(args.env_name, self.env)
        self.dataset = DatasetWrapper(Dataset(dataset_path, device=args.device),
                                 batch_size=args.batch_size,
                                 timestep=args.timestep, cls = self.frame_type)

        self.get_model()
        self.get_agent()

        self.get_api()

        self.get_renderer()
        self.vis = U.Visualizer(args.path)

        for i in range(args.num_epoch):
            print("TRAIN EPOCH", i)
            self.epoch(args.num_train_iter, args.num_valid_iter, num_eval=5, use_tqdm=True)

    def get_model(self):
        self.model = make_model(self.frame_type, self.args)

    def get_agent(self):
        self.agent = make_agent(self.model, self.frame_type, self.args).cuda(self.args.device)

    def get_renderer(self):
        self.renderer = Renderer(self.env)

    def get_api(self):
        self.rollout_predictor = RolloutWrapper(self.agent, self.frame_type)
        self.controller = RolloutCEM(self.rollout_predictor, self.env.action_space, iter_num=10,
                                     horizon=args.timestep-1, num_mutation=500, num_elite=50, device=args.device)

    def epoch(self, num_train, num_valid, num_eval=5, use_tqdm=False):

        def evaluate(to_vis, num_eval):
            to_vis['reward_eval'], trajectories = U.eval_policy(self.controller, self.env, eval_episodes=num_eval, save_video=0.,
                                                                video_path=os.path.join(self.path, "video{}.avi"), return_trajectories=True)

            obs, actions = trajectories[0]
            goal = obs[0]['desired_goal']
            obs = np.array([i['observation'] for i in obs]) # note that we always use goal conditioned environment
            actions = np.array(actions)

            as_frame = self.frame_type.from_observation(obs)
            predict = self.rollout_predictor.rollout(obs[None, 0], actions[None, :], None)[0].cpu()

            data = [as_frame[None, 0], actions, as_frame[None, 1:]]
            to_vis['rollout'] = self.renderer.render(data, {'predict': predict}, num=1, return_image=False, goal=goal)
            return to_vis


        ran = tqdm.trange if use_tqdm else range
        # train
        train_output = []
        for idx in ran(num_train):
            data = self.dataset.sample('train')

            self.agent.update_normalizer(data)
            info = self.agent.update(*data)
            train_output.append(info)
            if idx % 200 == 0:
                out = U.merge_training_output(train_output)
                if idx == 0:
                    out['image'] = self.renderer.render(data, info)
                self.vis(out)
                train_output = []

        # evaluate
        self.agent.eval()
        assert not self.agent.training
        valid_output = []
        to_vis = {}

        for i in ran(num_valid):
            data = self.dataset.sample('valid')
            info = self.agent.update(*data)
            valid_output.append(info)
            if i == num_valid - 1:
                to_vis = U.merge_training_output(valid_output)
                to_vis['valid_image'] = self.renderer.render(data, info)

        to_vis = {'valid_'+i:v for i, v in to_vis.items()}

        evaluate(to_vis, num_eval=num_eval)
        self.vis(to_vis, self.vis.tb.step - 1)

        torch.save(self.agent, os.path.join(self.path, 'agent'))
        self.agent.train()


if __name__ == '__main__':
    args = get_args()
    trainer(args)
