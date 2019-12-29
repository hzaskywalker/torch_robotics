import tqdm
import torch
import os
from robot.envs.arms.env import Env
from robot.model.gnn_forward import GNNForwardAgent, decode_state
import numpy as np
from robot.utils import togpu, train_loop


class Dataset:
    def __init__(self, env, num_trajs=1000, T=100, path='/dataset/tmp.pkl', batch_size=200, mode='state'):
        self.num_trajs = num_trajs
        self.T = T
        self.env = env

        if os.path.exists(path):
            tmp = torch.load(path)
        else:
            tmp = self.make_data()
            for i in tmp:
                print(i.shape)
            print('saving...')
            torch.save(tmp, path)
        self.obs, self.state, self.action, self.idx = tmp
        self.batch_size = batch_size
        self.mode = 'state'
        self.num_train = int(len(self.idx) * 0.8)


    def make_data(self):
        obs = []
        state = []
        action = []
        idx = []

        tot = 0

        for i in tqdm.trange(self.num_trajs):
            t = x = self.env.reset()
            for j in range(self.T):
                u = self.env.action_space.sample()
                t, r, d, _ = self.env.step(u)

                obs.append(x)
                state.append(self.env.encode(x))
                action.append(u)

                x = t
                idx.append(tot)
                tot += 1

            obs.append(t)
            state.append(self.env.encode(t))
            action.append(u)

        print('collected data')
        return np.array(obs), np.array(state), np.array(action), np.array(idx)


    def sample(self, mode='train', batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if mode == 'train':
            idx = np.random.choice(self.idx[:self.num_train], batch_size)
        else:
            idx = np.random.choice(self.idx[self.num_train:], batch_size)
        if self.mode == 'obs':
            out = [self.obs[idx], self.action[idx], self.obs[idx+1]]
        else:
            out = [self.state[idx], self.action[idx], self.state[idx+1]]
        return [togpu(i) for i in out]


def main():
    env = Env('arm3')
    network = 'mlp'
    mode = 'state'

    dataset = Dataset(env, num_trajs=1000)

    agent = GNNForwardAgent(network, 0.0001, env, layers=3, mid_channels=256).cuda()

    data = dataset.sample()
    agent.update(*data)

    agent = train_loop(agent, dataset, '/mp/forward_tmp/', 100000, 20, 1000, None, [], [], save_per_epsiode=False)



if __name__ == '__main__':
    main()
