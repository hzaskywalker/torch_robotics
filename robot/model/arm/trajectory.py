# only predict the next q
import os
import argparse
import tqdm
from robot.utils.models import fc
import numpy as np
from robot.utils.normalizer import Normalizer
from robot.utils.trainer import AgentBase, merge_training_output
from robot.utils.tensorboard import Visualizer
from robot.model.arm.dataset import Dataset
from robot.model.arm.recorder import gen_video, eval_policy
from robot.controller.pets.envs import make
import torch
from torch import nn


class INFO:
    inp_dim= (2 * 7, 7)
    oup_dim= (2 * 7, 3) # predict the end effector position...

    @classmethod
    def compute_reward(cls, s, a, ee, g):
        while len(g.shape) < len(ee.shape):
            g = g[None,:]
        return -(((ee[..., -3:]-g) ** 2).sum(dim=-1)) ** 0.5


class MLP(nn.Module):
    def __init__(self, inp_dim, oup_dim, num_layers, mid_channels, batchnorm=False):
        nn.Module.__init__(self)

        self.inp_dim = inp_dim
        self.oup_dim = oup_dim

        models = []
        cur = inp_dim
        for i in range(num_layers-1):
            models.append(fc(cur, mid_channels, relu=True, batch_norm=batchnorm))
            cur = mid_channels
        models.append(fc(cur, oup_dim, relu=False))
        self.main = nn.Sequential(*models)

    def forward(self, q):
        return self.main(q)

class MLP_ARM(nn.Module):
    def __init__(self, inp_dim, oup_dims, num_layers, mid_channels, batchnorm=False):
        nn.Module.__init__(self)

        self.q_dim = inp_dim[0]
        self.mlp1 = MLP(inp_dim[0] +inp_dim[1], oup_dims[0], num_layers, mid_channels, batchnorm=batchnorm)
        self.mlp2 = MLP(inp_dim[0], oup_dims[1], num_layers, mid_channels, batchnorm=batchnorm)

    def forward(self, state, action):
        new_state = state + self.mlp1(torch.cat((state, action), dim=-1)) # should we just use add here
        return new_state, self.mlp2(new_state)


class RolloutAgent(AgentBase):
    def __init__(self, model, lr, compute_reward):
        self.model = model
        super(RolloutAgent, self).__init__(model, lr)

        self.loss = nn.MSELoss()
        self.compute_reward = compute_reward

    def _rollout(self, s, a, goal=None):
        # s (inp_dim)
        # a (pop, T, acts)
        reward = 0
        states = []
        ees = []
        for i in range(a.shape[1]):
            t, ee = self.model(s, a[:, i])
            if goal is not None:
                reward = self.compute_reward(s, a, t, goal) + reward
            states.append(t)
            ees.append(ee)
            s = t
        return torch.stack(states, dim=1), torch.stack(ees, dim=1), reward

    def rollout(self, s, a, goal):
        assert not self.training
        with torch.no_grad():
            return self._rollout(s, a, goal)[-1]


    def update(self, state, actions, future, ee):
        if self.training:
            self.optim.zero_grad()
        dim = state.shape[-1]//2

        predict_future, ee_position, _ = self._rollout(state, actions, None)

        assert predict_future.shape == future.shape
        q_loss = self.loss(predict_future[...,:dim], future[...,:dim])
        dq_loss = self.loss(predict_future[...,dim:dim+dim], future[...,dim:dim+dim])
        assert ee_position.shape == ee.shape
        ee_loss = self.loss(ee_position, ee)

        if self.training:
            (q_loss + dq_loss + ee_loss).backward()

            torch.nn.utils.clip_grad_norm(self.model.parameters(), 1.) # to avoid exploision

            self.optim.step()
        return {
            'qloss': q_loss.detach().cpu().numpy(),
            'dqloss': dq_loss.detach().cpu().numpy(),
            'eeloss': ee_loss.detach().cpu().numpy(),
            'predict': torch.cat((predict_future, ee_position), dim=-1).detach().cpu().numpy()
        }

    def update_normalizer(self, batch):
        pass

class Tester:
    def __init__(self, env, make, path):
        self.env = env
        self.make = make
        self.path = path

        self.controller = RolloutCEM(self.model, self.env.action_space,
                                     iter_num=iter_num, horizon=horizon, num_mutation=num_mutation,
                                     num_elite=num_elite, device=self.model.device, **kwargs)

    def get_env(self):
        if isinstance(self.env, str):
            if self.make is None:
                self.env = make(self.env)
            else:
                self.env = self.make(self.env)
        return self.env

    def add_video(self, agent, to_vis):
        self.agent = agent
        to_vis['reward_eval'] = eval_policy(self, self.get_env(), eval_episodes=,
                                            save_video=1.,
                                            video_path=os.path.join(self.path, "video{}.avi"))
        to_vis['rollout'] = gen_video(self.get_env(), self, horizon=24)
        return to_vis

class Trainer:
    # pass
    # hard coding the training code here..
    def __init__(self, env, agent, dataset, batch_size, path, timestep=10, tester=None):
        self.env = env.unwrapped
        self.agent = agent
        self.dataset = dataset
        self.batch_size = batch_size
        self.path = path
        self.vis = Visualizer(path)

        dofs = self.env._actuator_dof['agent']
        self.dof_id = np.concatenate((dofs, dofs + 13))
        self.timestep = timestep
        self.tester = tester

    def sample_data(self, mode):
        data = self.dataset.sample(mode=mode, batch_size=self.batch_size, timestep=self.timestep, use_geom=False)

        all_states = data[0][:, :, self.dof_id]
        all_states[...,7:] *= 0.01 # times dt to make it consistent with state

        inp = all_states[:, 0]
        actions = data[1]

        future = all_states[:, 1:]
        ee = data[0][:, 1:, -3:]

        return inp, actions, future, ee

    def render_state(self, s, b=None):
        q = np.zeros((29,))
        q[self.dof_id[:7]] = s #only
        if b is not None:
            q[-3:] = b
        return self.env.unwrapped.render_state({'observation': q}, reset=False)

    def render(self, inp, oup):
        # TODO: how to visualize the velocity?
        if isinstance(inp[0], torch.Tensor):
            inp = [i.detach().cpu().numpy() for i in inp]

        state, _, future, ee = inp

        if isinstance(oup, torch.Tensor):
            oup = oup.detach().cpu().numpy()

        images = []
        for i in range(2):
            start = self.render_state(state[i][:7])
            ground_truth = [start] + [self.render_state(s[:7], ee) for s, ee in zip(future[i], ee[i])]
            predicted = [start]+ [self.render_state(o[:7], o[-3:]) for o in oup[i]]

            images.append(np.concatenate((
                np.concatenate(ground_truth, axis=1),
                np.concatenate(predicted, axis=1),
            ), axis=0))
        return np.stack(images)

    def epoch(self, num_train, num_valid, use_tqdm=False):
        ran = tqdm.trange if use_tqdm else range
        # train
        train_output = []
        for _ in ran(num_train):
            data = self.sample_data('train')

            self.agent.update_normalizer(data)
            info = self.agent.update(*data)
            train_output.append(info)
            if _ % 200 == 0:
                out = merge_training_output(train_output)
                if _ % 5000 == 4999:
                    out['image'] = self.render(data, info['predict'])
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
        to_vis = {'valid_'+i:v for i, v in to_vis.items()}
        to_vis['valid_image'] = self.render(data, info['predict'])

        if self.tester is not None:
            self.tester.add_video(self, to_vis)

        self.vis(to_vis, self.vis.tb.step - 1)

        self.agent.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchnorm", type=int, default=0)
    parser.add_argument("--model", type=str, default='mlp')
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--num_epoch", type=int, default=1000)
    parser.add_argument("--num_train_iter", type=int, default=10000)
    parser.add_argument("--num_valid_iter", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--path", type=str, default='rollout')
    parser.add_argument("--timestep", type=int, default=10)
    args = parser.parse_args()

    info = INFO()

    if args.model == 'mlp':
        model = MLP_ARM(info.inp_dim, info.oup_dim, 4, 256, batchnorm=args.batchnorm)
    else:
        raise NotImplementedError

    agent = RolloutAgent(model, lr=args.lr, compute_reward=info.compute_reward).cuda()

    dataset = Dataset('/dataset/arm')
    env, env_params = make('armreach')

    trainer = Trainer(env, agent, dataset, batch_size=args.batch_size, path=args.path, timestep=args.timestep)

    for i in range(args.num_epoch):
        print("TRAIN EPOCH", i)
        trainer.epoch(args.num_train_iter, args.num_valid_iter, use_tqdm=True)


if __name__ == '__main__':
    main()
