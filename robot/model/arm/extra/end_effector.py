# just learning the end-effector position
# this is a pure vision problem. If the network can't do frame transformation easily, there is no way that it can do this.
# maybe we can consider to design a coordinate transformation network to facilate it.
import argparse
import tqdm
from robot.utils.models import fc
import numpy as np
from robot.utils.normalizer import Normalizer
from robot.utils.trainer import AgentBase, merge_training_output
from robot.utils.tensorboard import Visualizer
from robot.model.arm.dataset import Dataset
from robot.controller.pets.envs import make
import torch
from torch import nn


class INFO:
    inp_dim=7
    oup_dim=3


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


class EndEffectorAgent(AgentBase):
    def __init__(self, model, lr, normalizer=None):
        self.model = model
        self.normalizer = normalizer
        super(EndEffectorAgent, self).__init__(model, lr)

        self.loss = nn.MSELoss()

    def predict(self, inp):
        with torch.no_grad():
            training = self.training
            self.eval()
            if self.normalizer is not None:
                inp = self.normalizer(inp)

            output =  self.model(inp).detach().cpu().numpy()
            if training:
                self.train()
            return output

    def update(self, inp, target):
        if self.training:
            self.optim.zero_grad()

        if self.normalizer is not None:
            inp = self.normalizer(inp)

        output = self.model(inp)
        assert output.shape == target.shape
        loss = self.loss(output, target)

        if self.training:
            loss.backward()
            self.optim.step()
        return {
            'loss': loss.detach().cpu().numpy(),
            'predict': output.detach().cpu().numpy()
        }

    def update_normalizer(self, batch):
        if self.normalizer is not None:
            self.normalizer.update(batch[0]) # update input


class Trainer:
    # pass
    def __init__(self, env, agent, dataset, batch_size, path):
        self.env = env.unwrapped
        self.agent = agent
        self.dataset = dataset
        self.batch_size = batch_size
        self.path = path
        self.vis = Visualizer(path)

        self.dof_id = self.env._actuator_dof['agent']

    def sample_data(self, mode):
        data = self.dataset.sample(mode=mode, batch_size=self.batch_size, timestep=1, use_geom=False)
        x = data[0][:, 0]
        # this should absolutely wrong...
        return x[:, self.dof_id], x[:, -3:]

    def render(self, inp, oup):
        if isinstance(inp, torch.Tensor):
            inp = inp.detach().cpu().numpy()
        if isinstance(oup, torch.Tensor):
            oup = oup.detach().cpu().numpy()
        images = []
        for a, b in zip(inp, oup):
            q = np.zeros((29,))
            q[self.dof_id] = a
            q[-3:] = b
            images.append(self.env.unwrapped.render_obs({'observation':q}))
            if len(images) > 10:
                break
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
                if _ % 2000 == 0:
                    out['image'] = self.render(data[0], info['predict'])
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
        to_vis['valid_image'] = self.render(data[0], info['predict'])
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
    parser.add_argument("--path", type=str, default='tmp')
    args = parser.parse_args()

    info = INFO()

    if args.model == 'mlp':
        model = MLP(info.inp_dim, info.oup_dim, 4, 256, batchnorm=args.batchnorm)
    else:
        raise NotImplementedError

    agent = EndEffectorAgent(model, lr=args.lr, normalizer=None).cuda()

    dataset = Dataset('/dataset/arm')
    env, _ = make('armreach')

    trainer = Trainer(env, agent, dataset, batch_size=args.batch_size, path=args.path)

    for i in range(args.num_epoch):
        print("TRAIN EPOCH", i)
        trainer.epoch(args.num_train_iter, args.num_valid_iter, use_tqdm=True)


if __name__ == '__main__':
    main()
