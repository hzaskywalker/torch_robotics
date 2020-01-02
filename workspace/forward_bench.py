## evaluate of the forward model
import os
import numpy as np
import torch
from robot.utils import rollout, cache, as_input, batch_runner, rot6d


DATASET = {
    'Pendulum': '/tmp/pendulum_test'
}


class ForwardModelTester:
    def __init__(self, env, path=None, num_traj=100, timestep=1000):
        self.path = path
        self.state_format = env.state_format

        @cache(self.path)
        def collect_data(env, num_traj, timestep):
            trajs = []
            for i in range(num_traj):
                trajs.append(rollout(env, timestep=timestep))
            return trajs

        self.trajs = collect_data(env, num_traj, timestep)

    def make_test_data(self, t=1):
        #for i in self.trajs:
        S, A, T = [], [], []
        for s, a in self.trajs:
            S.append(s[:-t])
            A.append(a[:-t])
            T.append(s[t:])
        return np.concatenate(S), np.concatenate(A), np.concatenate(T)

    def dist(self, state, gt):
        if len(state.shape) == 2:
            # make it into nodes
            state = self.state_format.decode(state)[0]

        if len(gt.shape) == 2:
            gt = self.state_format.decode(gt)[0]
        state = torch.Tensor(state)
        gt = torch.Tensor(gt)

        x = slice(3)
        w = slice(3, 9)
        dw = slice(9, 12)
        dx = slice(12, 15)
        assert state.shape == gt.shape, f"state.shape {state.shape} gt.shape {gt.shape}"
        out = (((state[..., x] - gt[..., x])**2).sum(dim=-1),
               rot6d.rdist(state[..., w], gt[..., w]),
               ((state[..., dw] - gt[..., dw]) ** 2).sum(dim=-1),
               ((state[..., dx] - gt[..., dx]) ** 2).sum(dim=-1))
        return torch.stack(out, dim=-1)

    def eval(self, s, t, predict):
        # estimate normalized error of velolicy, acceleration
        d_pre = self.dist(predict, t).mean(dim=(0,1))
        s_pre = self.dist(s, t).mean(dim=(0, 1))
        return (d_pre/s_pre.clamp(1e-6, float('inf'))).mean()

    def test(self, agent, visualize=False):
        f = as_input(2)(batch_runner(128, show=False)(agent))
        s, a, t = self.make_test_data()
        predict = f(s, a)
        return self.eval(s, t, predict)



def test():
    dataset = 'Pendulum'
    from gnn import make
    path = DATASET[dataset]

    env = make(dataset)
    tester = ForwardModelTester(env, path, 10, 1000)

    model_path = '/tmp/pendulum/agent'
    agent = torch.load(model_path)

    print(tester.test(agent))

if __name__ == '__main__':
    test()
