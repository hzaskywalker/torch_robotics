import torch
import numpy as np
from torch import nn
from robot import A, U


class AngleLoss(nn.Module):
    def forward(self, predict, label):
        diff = torch.abs(predict - label)
        return (torch.min(diff, 2 * np.pi - diff) ** 2).mean()


class AcrobatFrame(A.ArmBase):
    dim = 2
    d_ee = 2
    max_dq = 20
    input_dims = (dim, dim, dim)
    output_dims = d_ee # output the ddq and the position of the end-effectors..

    # action factor, the observed_action * action_norm <= max_a
    action_norm = 1
    max_a = 1

    angle_loss = AngleLoss()

    def as_input(self, action=None):
        action = action.clamp(-self.max_a, self.max_a)
        return torch.cat((self.q, self.dq), dim=-1), action

    def as_observation(self):
        s = self.q
        assert s.shape[0] == self.dim, f"ERROOR: s.shape: {s.shape}"
        q = np.zeros((*s.shape[:-1], self.dim * 2 + self.d_ee))

        q[:self.dim] = U.tocpu(s)
        if self.ee is not None:
            q[-self.d_ee:] = U.tocpu(self.ee)
        return {'observation': q}

    @classmethod
    def from_observation(cls, observation):
        q = observation[..., :cls.dim]
        dq = observation[..., cls.dim:cls.dim + cls.dim]
        ee = observation[..., -cls.d_ee:]
        return cls(q, dq, ee)

    def add(self, new_state, ee):
        q = (new_state[...,:self.dim] + np.pi) % (2 * np.pi) - np.pi# we don't clip the q as we will use sin/cos as the input
        dq = new_state[..., self.dim:self.dim * 2].clamp(-self.max_dq, self.max_dq)
        return self.__class__(q, dq, ee)

    def calc_loss(self, label):
        # label are also type a frame..
        assert self.q.shape == label.q.shape
        assert self.dq.shape == label.dq.shape
        assert self.ee.shape == label.ee.shape
        return {
            'q_loss': self.angle_loss(self.q, label.q),
            'dq_loss': self.loss(self.dq, label.dq),
            'ee_loss': self.loss(self.ee, label.ee)
        }


class SapienAcrobat2Frame(AcrobatFrame):
    def as_observation(self):
        s = self.q
        assert s.shape[0] == self.dim, f"ERROOR: s.shape: {s.shape}"
        q = np.zeros((*s.shape[:-1], self.dim * 3 + self.d_ee))

        q[:self.dim] = U.tocpu(s)
        if self.ee is not None:
            q[-self.d_ee:] = U.tocpu(self.ee)
        return {'observation': q}

    def calc_loss(self, label):
        # label are also type a frame..
        assert self.q.shape == label.q.shape
        assert self.dq.shape == label.dq.shape
        assert self.ee.shape == label.ee.shape
        return {
            'q_loss': self.angle_loss(self.q, label.q),
            'dq_loss': self.loss(self.dq, label.dq) * 0.001,
            'ee_loss': self.loss(self.ee, label.ee)
        }


FRAMETYPES={
    'diff_acrobat2': AcrobatFrame,
    'acrobat2': SapienAcrobat2Frame
}
