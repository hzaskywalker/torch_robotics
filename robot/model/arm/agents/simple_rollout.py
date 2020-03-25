from torch import nn
import torch
from robot.utils.trainer import AgentBase

class RolloutAgent(AgentBase):
    def __init__(self, model, lr, encode_obs, compute_reward,
                 max_q=None, max_dq=None, max_a=None, weight_q=1., weight_dq=1., weight_ee=1.):
        self.model = model
        super(RolloutAgent, self).__init__(model, lr)

        self.loss = nn.MSELoss()

        self.compute_reward = compute_reward
        self.encode_obs = encode_obs # transforms obs into

        # the maximum angle can only be 10
        self.max_q = max_q
        self.max_dq = max_dq
        self.max_a = max_a
        print('max q:', self.max_q)
        print('max dq:', self.max_dq)
        print('max a:', self.max_a)
        assert self.max_a == 1, "currently we only support known action set"

        self.weight_q = weight_q
        self.weight_dq = weight_dq
        self.weight_ee = weight_ee

    def _rollout(self, s, a, goal=None):
        # s (inp_dim)
        # a (pop, T, acts)
        states, ees, reward = [], [], 0
        dim_q = s.shape[-1]//2
        # do clamp
        if self.max_a is not None:
            a = a.clamp(-self.max_a, self.max_a)
        for i in range(a.shape[1]):
            if self.max_q is not None and self.max_dq is not None:
                s = torch.cat((s[...,:dim_q].clamp(-self.max_q, self.max_q),
                               s[...,dim_q:].clamp(-self.max_dq, self.max_dq)), dim=-1)

            t, ee = self.model(s, a[:, i])
            if goal is not None:
                reward = self.compute_reward(s, a, t, ee, goal) + reward
            states.append(t)
            ees.append(ee)
            s = t

        # return cost for optimization
        return torch.stack(states, dim=1), torch.stack(ees, dim=1), -reward


    def rollout(self, obs, a, goal=None, return_traj=False):
        # notice that the input is the observation ..
        s, _ = self.encode_obs(obs) # first encode it..
        assert not self.training

        with torch.no_grad():
            if not return_traj:
                return None, self._rollout(s, a, goal)[-1]
            else:
                return self._rollout(s, a, goal)[:2]


    def update(self, state, actions, future, ee):
        if self.training:
            self.optim.zero_grad()
        dim = state.shape[-1]//2

        predict_future, predict_ee, _ = self._rollout(state, actions, None)

        assert predict_future.shape == future.shape
        q_loss = self.loss(predict_future[...,:dim], future[...,:dim])
        dq_loss = self.loss(predict_future[...,dim:dim+dim], future[...,dim:dim+dim])
        assert predict_ee.shape == ee.shape
        ee_loss = self.loss(predict_ee, ee)

        if self.training:
            (q_loss * self.weight_q + dq_loss * self.weight_dq + ee_loss * self.weight_ee).backward()
            self.optim.step()

        return {
            'qloss': q_loss.detach().cpu().numpy(),
            'dqloss': dq_loss.detach().cpu().numpy(),
            'eeloss': ee_loss.detach().cpu().numpy(),
            'predict_future': predict_future.detach().cpu().numpy(),
            'predict_ee': predict_ee.detach().cpu().numpy(),
        }


    def update_normalizer(self, batch):
        pass
