# only predict the next q
import os
import argparse
import tqdm
from robot.utils.models import fc
import numpy as np
from robot.utils.normalizer import Normalizer
from robot.utils import tocpu
from robot.utils.trainer import AgentBase, merge_training_output
from robot.utils.tensorboard import Visualizer
from robot.model.arm.dataset import Dataset
from robot.model.arm.recorder import gen_video, eval_policy
from robot.controller.pets.envs import make
import torch
from torch import nn


def make_info(env):
    env = env.unwrapped
    dofs = env._actuator_dof['agent']

    class INFO:
        inp_dim= (2 * 7, 7)
        oup_dim= (2 * 7, 3) # predict the end effector position...
        dof_id = np.concatenate((dofs, dofs + 13))

        @classmethod
        def compute_reward(cls, s, a, ee, g):
            while len(g.shape) < len(ee.shape):
                g = g[None,:]
            return -(((ee[..., -3:]-g) ** 2).sum(dim=-1)) ** 0.5

        @classmethod
        def encode_obs(cls, s):
            s = s[..., cls.dof_id]
            assert len(cls.dof_id) == 14
            s[..., 7:] *= 0.01  # times dt to make it consistent with state
            return s

    return INFO


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
        self.mlp2 = MLP(inp_dim[0]//2, oup_dims[1], num_layers, mid_channels, batchnorm=batchnorm)

    def forward(self, state, action):
        new_state = state + self.mlp1(torch.cat((state, action), dim=-1)) # should we just use add here
        return new_state, self.mlp2(new_state[...,:7])


class RolloutAgent(AgentBase):
    def __init__(self, model, lr, compute_reward):
        self.model = model
        super(RolloutAgent, self).__init__(model, lr)

        self.loss = nn.MSELoss()
        self.compute_reward = compute_reward

        # the maximum angle can only be 10
        self.max_q = 7
        # maximum velocity is limited to 2000, but note we change the meaning of dq by timing it with 0.01
        self.max_dq = 2000 * 0.01
        self.max_a = 1

    def _rollout(self, s, a, goal=None):
        # s (inp_dim)
        # a (pop, T, acts)
        states, ees, reward = [], [], 0
        dim = s.shape[-1]//2
        # do clamp
        a = a.clamp(-self.max_a, self.max_a)
        for i in range(a.shape[1]):
            s = torch.cat((s[...,:dim].clamp(-self.max_q, self.max_q),
                           s[...,dim:].clamp(-self.max_dq, self.max_dq)), dim=-1)
            #s = s.clamp(-self.max_dq, self.max_dq)

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
            self.optim.step()

        return {
            'qloss': q_loss.detach().cpu().numpy(),
            'dqloss': dq_loss.detach().cpu().numpy(),
            'eeloss': ee_loss.detach().cpu().numpy(),
            'predict': torch.cat((predict_future, ee_position), dim=-1).detach().cpu().numpy()
        }

    def update_normalizer(self, batch):
        pass


def render_state(env, s, b=None):
    q = np.zeros((29,))
    # TODO: remove the hack here
    q[np.arange(7)+1] = s #only
    if b is not None:
        q[-3:] = b
    return env.unwrapped.render_state({'observation': q}, reset=False)

def render(env, inp, oup):
    # TODO: how to visualize the velocity?
    if isinstance(inp[0], torch.Tensor):
        inp = [i.detach().cpu().numpy() for i in inp]

    state, _, future, ee = inp

    if isinstance(oup, torch.Tensor):
        oup = oup.detach().cpu().numpy()

    images = []
    for i in range(2):
        start = render_state(env, state[i][:7])
        ground_truth = [start] + [render_state(env, s[:7], ee) for s, ee in zip(future[i], ee[i])]
        predicted = [start]+ [render_state(env, o[:7], o[-3:]) for o in oup[i]]

        images.append(np.concatenate((
            np.concatenate(ground_truth, axis=1),
            np.concatenate(predicted, axis=1),
        ), axis=0))
    return np.stack(images)


class Tester:
    def __init__(self, env, agent, path, encode_obs, horizon, iter_num, num_mutation, num_elite, device, **kwargs):
        self.env = env
        self.path = path

        from robot.controller.pets.planner import RolloutCEM

        self.agent = agent
        self.device = device

        self.encode_obs = encode_obs
        self.controller = RolloutCEM(self.agent, self.env.action_space,
                                     iter_num=iter_num, horizon=horizon, num_mutation=num_mutation,
                                     num_elite=num_elite, device=device, **kwargs)

    def reset(self):
        self.controller.reset()

    def __call__(self, observation):
        x = observation['observation']
        goal = observation['desired_goal']
        x = self.encode_obs(torch.tensor(x, dtype=torch.float, device=self.device))
        goal = torch.tensor(goal, dtype=torch.float, device=self.device)
        out = tocpu(self.controller(x, goal))
        return out

    def add_video(self, agent, to_vis):
        self.agent = agent
        to_vis['reward_eval'] = eval_policy(self, self.env, eval_episodes=5,
                                            save_video=1., video_path=os.path.join(self.path, "video{}.avi"))
        to_vis['rollout'] = self.gen_video(horizon=24)
        return to_vis

    def gen_video(self, horizon=24):
        import torch
        # write the video at the neighborhood of the optimal [random] policy
        env = self.env
        start = obs = env.reset()
        real_trajs = []
        actions = []
        # 100 frame
        for i in range(horizon):
            action = self(obs)
            actions.append(action)
            real_trajs.append(env.unwrapped.render_state(obs))
            obs = env.step(action)[0]

        fake_trajs = []
        obs = start

        s = torch.tensor(obs['observation'], dtype=torch.float32, device=self.device)
        a = torch.tensor(actions, dtype=torch.float32, device=self.device)
        state, ee = self.agent._rollout(self.encode_obs(s[None,:]), a[None,:])[:2]

        state = state[0].detach().cpu().numpy()
        ee = ee[0].detach().cpu().numpy()

        fake_trajs.append(env.unwrapped.render_state(obs))
        for s, e in zip(state, ee):
            fake_trajs.append(render_state(env, s[:7], e))

        for a, b in zip(real_trajs, fake_trajs):
            yield np.concatenate((a, b), axis=1)


class Trainer:
    # pass
    # hard coding the training code here..
    def __init__(self, env, agent, dataset, batch_size, path, encode_obs, timestep=10, tester=None):
        self.env = env.unwrapped
        self.agent = agent
        self.dataset = dataset
        self.batch_size = batch_size
        self.path = path
        self.vis = Visualizer(path)

        self.timestep = timestep
        self.tester = tester
        self.encode_obs = encode_obs

    def sample_data(self, mode):
        data = self.dataset.sample(mode=mode, batch_size=self.batch_size, timestep=self.timestep, use_geom=False)

        all_states = self.encode_obs(data[0])

        inp = all_states[:, 0]
        actions = data[1]

        future = all_states[:, 1:]
        ee = data[0][:, 1:, -3:]

        return inp, actions, future, ee

    def epoch(self, num_train, num_valid, use_tqdm=False):
        ran = tqdm.trange if use_tqdm else range
        # train
        train_output = []
        cc = 0
        for idx in ran(num_train):
            data = self.sample_data('train')

            self.agent.update_normalizer(data)
            info = self.agent.update(*data)
            train_output.append(info)
            if idx % 200 == 199:
                out = merge_training_output(train_output)
                cc += 1
                if cc % 25 == 0:
                    out['image'] = render(self.env, data, info['predict'])
                    print(out['image'].shape)

                #self.agent.eval()
                #self.tester.add_video(self.agent, out)
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
        to_vis['valid_image'] = render(self.env, data, info['predict'])

        if self.tester is not None:
            self.tester.add_video(self.agent, to_vis)

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

    env, env_params = make('armreach')
    info = make_info(env)

    if args.model == 'mlp':
        model = MLP_ARM(info.inp_dim, info.oup_dim, 4, 256, batchnorm=args.batchnorm)
    else:
        raise NotImplementedError

    agent = RolloutAgent(model, lr=args.lr, compute_reward=info.compute_reward).cuda()

    dataset = Dataset('/dataset/arm')

    tester = Tester(env, agent, args.path, encode_obs=info.encode_obs,
                    horizon=args.timestep-1, iter_num=10, num_mutation=500, num_elite=50, device='cuda:0')
    trainer = Trainer(env, agent, dataset, encode_obs=info.encode_obs,
                      batch_size=args.batch_size, path=args.path, timestep=args.timestep, tester=tester)

    for i in range(args.num_epoch):
        print("TRAIN EPOCH", i)
        trainer.epoch(args.num_train_iter, args.num_valid_iter, use_tqdm=True)


if __name__ == '__main__':
    main()
