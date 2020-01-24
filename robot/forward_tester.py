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
        self.extension = env.extension
        self.render_state = env.render_state

        @cache(self.path)
        def collect_data(env, num_traj, timestep):
            trajs = []
            for i in range(num_traj):
                trajs.append(rollout(env, timestep=timestep)[:2])
            return trajs

        self.trajs = collect_data(env, num_traj, timestep)

    def make_test_data(self, t=1):
        #for i in self.trajs:
        S, A, T = [], [], []
        for s, a in self.trajs:
            S.append(s[:-t])

            action = np.stack([a[i:-t+i] for i in range(t)], axis=1)
            A.append(action)
            T.append(s[t:])
        return np.concatenate(S), np.concatenate(A), np.concatenate(T)

    def dist(self, state, gt):
        state = torch.Tensor(state)
        gt = torch.Tensor(gt)

        x = slice(3)
        dx = slice(3, 6)
        w = slice(6, 12)
        dw = slice(12, 15)
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

    def render(self, agent, t=20):
        s = np.array(self.trajs[0][0][0])
        a = np.array(self.trajs[0][1][:t])
        ag = as_input(2)(agent)

        idx = 0
        yield np.concatenate(
            (self.render_state(self.extension.decode(s)[0]),
             self.render_state(self.extension.decode(self.trajs[0][0][idx])[0])), axis=0)
        for i in a:
            s = ag(s, i)
            idx += 1
            yield np.concatenate(
                (self.render_state(self.extension.decode(s)[0]),
                 self.render_state(self.extension.decode(self.trajs[0][0][idx])[0])), axis=0)

    def test(self, agent, t=1):
        f = as_input(2)(batch_runner(128, show=False)(agent.rollout))
        s, a, t = self.make_test_data(t)
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

    f = tester.render(agent, t=100)
    #for t in [1, 10, 100]:
    #    print(t, tester.test(agent, t=t))
    from robot.utils import write_video
    write_video(f, 'xx.avi')

if __name__ == '__main__':
    test()
