import torch
import numpy as np
import time
from robot.utils import AgentBase
from robot.utils.data_parallel import DataParallel

class Rollout:
    def __init__(self, make, env_name):
        self.env = make(env_name)

    def __call__(self, s, a):
        d = len(self.env.init_qpos)
        trajs = []
        rewards = []
        for s, a in zip(s, a):
            s[0] = 0
            self.env.set_state(s[:d], s[d:])

            is_forward = len(a.shape) == 1
            if is_forward:
                a = a[None,:]

            reward = 0
            tmp = []
            for action in a:
                t = self.env.step(action)[0]
                reward = reward + self.env.state_prior.cost(
                    torch.Tensor(s[None,:]), torch.Tensor(action[None,:]), torch.Tensor(t[None,:]))[0].detach().cpu().numpy()
                tmp.append(t)
                s = t

            if is_forward:
                tmp = tmp[0]

            trajs.append(tmp)
            rewards.append(reward)
        return np.array(trajs), np.array(rewards)

    @staticmethod
    def test():
        from robot.envs.gym import make
        print('make rollout')
        env_name = 'MBRLCartpole-v0'
        rollout = Rollout(make, env_name)

        env = make(env_name)
        obs = np.array([env.reset() for i in range(10)])
        acts = np.array([[env.action_space.sample() for j in range(5)]for i in range(10)])

        outs, rewards = rollout(obs, acts)
        print(outs.shape, rewards.shape)

        outs, rewards = rollout(obs, acts[:, 0])
        print(outs.shape, rewards.shape)




class GTModel(AgentBase):
    def __init__(self, make, env_name, num_process=10):
        assert num_process >= 1
        self.rollout = DataParallel(num_process, Rollout, make, env_name)

    def update(self, *args, **kwargs):
        pass

    def cuda(self):
        pass

    def reset(self):
        pass

    def save(self):
        pass

    def __call__(self, s, a):
        return self.forward(s, a)

    def forward(self, s, a):
        # transform to numpy
        is_np = not isinstance(s, torch.Tensor)
        if not is_np:
            _d = s.device
            s = s.cpu().detach().numpy()
            a = a.cpu().detach().numpy()

        # transform to batch
        is_batch = (len(s.shape) > 1)
        if not is_batch:
            s = [s]
            a = [a]

        outs, rewards = self.rollout(np.array(s), np.array(a))

        if not is_np:
            outs = torch.Tensor(outs).to(_d)
            rewards = torch.Tensor(rewards).to(_d)
        if not is_batch:
            outs = outs[0]
            rewards = rewards[0]

        return outs, rewards

    def rollout(self, s, a):
        return self(s, a)

    @staticmethod
    def test():
        pass



if __name__ == '__main__':
    Rollout.test()