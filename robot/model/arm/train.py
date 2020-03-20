# only predict the next q
import os
import argparse
import tqdm
import numpy as np
from robot.utils import tocpu
from robot.utils.trainer import merge_training_output
from robot.utils.rl_utils import eval_policy
from robot.utils.tensorboard import Visualizer
from robot.controller.rollout_controller import RolloutCEM

import torch
from robot.model.arm.dataset import Dataset
from robot.model.arm.envs import make
from robot.model.arm.config import make_info
from robot.model.arm.models import make_model
from robot.model.arm.agents import make_agent
from robot.model.arm.renderer import Renderer


class Trainer:
    # pass
    # hard coding the training code here..
    def __init__(self, env, agent, dataset, batch_size, path, encode_obs,
                 timestep=10, controller=None, renderer=None):
        self.env = env
        self.agent = agent
        self.dataset = dataset
        self.batch_size = batch_size
        self.path = path
        self.vis = Visualizer(path)

        self.timestep = timestep
        self.controller = controller
        self.renderer = renderer
        self.encode_obs = encode_obs

    def sample_data(self, mode):
        observation, actions = self.dataset.sample(mode=mode, batch_size=self.batch_size, timestep=self.timestep, use_geom=False)

        all_states, ee = self.encode_obs(observation)
        inp = all_states[:, 0]
        future = all_states[:, 1:]
        ee = ee[:, 1:]
        return inp, actions, future, ee

    def render_image(self, data, info):
        return self.renderer.render_image(data[0], data[2], data[3], info['predict_future'], info['predict_ee'])

    def add_video(self, to_vis, num_eval):
        to_vis['reward_eval'] = eval_policy(self.controller, self.env, eval_episodes=num_eval,
                                            save_video=1., video_path=os.path.join(self.path, "video{}.avi"))
        to_vis['rollout'] = self.renderer.render_video(self.controller, self.agent, horizon=24)
        return to_vis

    def epoch(self, num_train, num_valid, num_eval=5, use_tqdm=False):
        ran = tqdm.trange if use_tqdm else range
        # train
        train_output = []
        cc = 0
        for idx in ran(num_train):
            data = self.sample_data('train')

            self.agent.update_normalizer(data)
            info = self.agent.update(*data)
            train_output.append(info)
            out = None
            if idx % 200 == 199:
                out = merge_training_output(train_output)
                cc += 1
                if cc % 25 == 1:
                    out['image'] = self.render_image(data, info)
                self.vis(out)
                train_output = []

        # evaluate
        self.agent.eval()
        assert not self.agent.training
        valid_output = []
        for _ in ran(num_valid):
            data = self.sample_data('valid')
            info = self.agent.update(*data)
            valid_output.append(info)

        to_vis = merge_training_output(valid_output)
        if num_valid > 0:
            to_vis['valid_image'] = self.render_image(data, info)
        to_vis = {'valid_'+i:v for i, v in to_vis.items()}

        self.add_video(to_vis, num_eval=num_eval)
        self.vis(to_vis, self.vis.tb.step - 1)

        torch.save(self.agent, os.path.join(self.path, 'agent'))

        self.agent.train()


def main():
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

    env = make(args.env_name)
    dataset_path = os.path.join('/dataset/', args.env_name)
    info = make_info(args.env_name, env)

    model = make_model(info, args)
    agent = make_agent(model, info, args).cuda(args.device)
    dataset = Dataset(dataset_path, device=args.device)


    # controller
    controller = RolloutCEM(agent, env.action_space, iter_num=10, horizon=args.timestep-1, num_mutation=500,
                                 num_elite=50, device=args.device)

    renderer = Renderer(args.env_name, env)

    trainer = Trainer(env, agent, dataset, encode_obs=info.encode_obs,
                      batch_size=args.batch_size, path=args.path, timestep=args.timestep,
                      controller=controller, renderer=renderer)

    for i in range(args.num_epoch):
        print("TRAIN EPOCH", i)
        trainer.epoch(args.num_train_iter, args.num_valid_iter, num_eval=5, use_tqdm=True)


if __name__ == '__main__':
    main()
