# only predict the next q
import os
import argparse
import tqdm
import numpy as np
import pickle
from robot.controller.rollout_controller import RolloutCEM

import torch
from robot.model.arm.dataset import Dataset
from robot.model.arm.envs import make
from robot import U
from .agents import RolloutAgent


class DatasetWrapper:
    def __init__(self, dataset, batch_size, timestep, cls):
        self.dataset = dataset
        self.batch_size = batch_size
        self.timestep = timestep
        self.cls = cls

    def sample(self, mode):
        obs, action = self.dataset.sample(mode=mode, batch_size=self.batch_size, timestep=self.timestep)
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
    parser.add_argument("--env_name", type=str, default=None, choices=['arm', 'acrobat2', 'plane', 'cheetah'])
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
    parser.add_argument("--weight_dq", type=float, default=1)
    parser.add_argument("--weight_ee", type=float, default=1.)
    parser.add_argument("--resume", type=int, default=0)
    args = parser.parse_args()
    return args


class trainer:
    def __init__(self, args):
        self.args = args
        self.path = self.args.path
        self.vis = U.Visualizer(args.path)

        with open(os.path.join(args.path, 'args'), 'wb') as f:
            pickle.dump(args, f)

        self.set_env()
        self.set_model()
        self.set_agent()

        self.set_policy()

        self.set_renderer()

        self.epoch_num = 0
        for self.epoch_num in range(args.num_epoch):
            self.epoch(args.num_train_iter, args.num_valid_iter, num_eval=5, use_tqdm=True)
            # we save the agent by default
            self.save()

    def make_frame_cls(self, env_name, env):
        from .frame import ArmBase, Plane
        if env_name == 'arm':
            return ArmBase
        elif env_name == 'plane':
            return Plane
        else:
            raise NotImplementedError

    def set_env(self):
        args = self.args
        self.env = make(args.env_name)
        dataset_path = os.path.join('/dataset/', args.env_name)
        self.frame_type = self.make_frame_cls(args.env_name, self.env)
        self.dataset = DatasetWrapper(Dataset(dataset_path, device=args.device),
                                      batch_size=args.batch_size,
                                      timestep=args.timestep, cls = self.frame_type)

    def set_model(self):
        from .models import MLP_ARM
        cls = self.frame_type
        self.model = MLP_ARM(cls.input_dims, cls.output_dims, 4, 256, batchnorm=self.args.batchnorm)

    def set_agent(self):
        # max_a is very important to make it work...
        args = self.args
        if not args.resume:
            agent = RolloutAgent(self.model, lr=args.lr, loss_weights={
                'q_loss': args.weight_q,
                'dq_loss': args.weight_dq,
                'ee_loss': args.weight_ee,
            })
        else:
            import torch, os
            agent = torch.load(os.path.join(args.path, 'agent'))

        self.agent = agent.cuda()

    def set_renderer(self):
        self.renderer = Renderer(self.env)

    def set_rollout_model(self):
        self.rollout_predictor = RolloutWrapper(self.agent, self.frame_type)

    def set_policy(self):
        self.set_rollout_model()
        args = self.args
        self.controller = RolloutCEM(self.rollout_predictor, self.env.action_space, iter_num=10,
                                     horizon=args.timestep-1, num_mutation=500, num_elite=50, device=args.device)

    def epoch(self, num_train, num_valid, num_eval=5, use_tqdm=False):
        print("TRAIN EPOCH", self.epoch_num)

        def evaluate(to_vis, num_eval):
            if num_eval == 0:
                return

            policy = U.eval_policy(self.controller, self.env, eval_episodes=num_eval, save_video=0.,
                                   video_path=os.path.join(self.path, "video{}.avi"), return_trajectories=True)
            to_vis['reward_eval'], trajectories = policy

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
        self.agent.train()

    def save(self):
        torch.save(self.agent, os.path.join(self.path, 'agent'))
