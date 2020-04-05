# the most simple model of the acrobat
# the input is transformed into sin/cos
import torch
from torch import nn
import numpy as np
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
        # make sure observation is a tensor
        # this is indeed the deserialization ...
        assert observation.shape[-1] == cls.dim *2 + cls.d_ee
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


class MLP_ACROBAT(nn.Module):
    def __init__(self, inp_dim, oup_dims, num_layers, mid_channels, batchnorm=False):
        nn.Module.__init__(self)

        self.dof = inp_dim[0]
        self.mlp1 = A.models.MLP(inp_dim[0] * 2+ inp_dim[1] + inp_dim[2],
                                 inp_dim[0] + inp_dim[1], num_layers, mid_channels, batchnorm=batchnorm)

        self.mlp2 = A.models.MLP(inp_dim[0] * 2, oup_dims, num_layers, mid_channels, batchnorm=batchnorm)

    def wrap(self, q):
        # the wrapper is the information of the neural network
        return torch.cat((torch.sin(q), torch.cos(q)), dim=-1)

    def forward(self, state, action):
        q, dq = state[..., :self.dof], state[..., self.dof:]

        inp = torch.cat((self.wrap(q), dq, action), dim=-1)
        delta = self.mlp1(inp) # should we just use add here
        new_q = q + delta[..., :self.dof]
        new_dq = dq + delta[..., self.dof:]
        return torch.cat((new_q, new_dq), dim=-1), self.mlp2(self.wrap(new_q))


class AcrobatTrainer(A.trainer):
    def __init__(self, args):
        args.env_name = 'diff_acrobat'
        args.num_train_iter = 20000
        args.num_valid_iter = 20
        args.timestep = 2
        args.lr = 0.01
        super(AcrobatTrainer, self).__init__(args)

    def set_model(self):
        from robot.model.arm.acrobat.phys_model import ArmModel
        #self.model = MLP_ACROBAT(self.frame_type.input_dims, self.frame_type.output_dims,
        #                         4, 256, batchnorm=self.args.batchnorm)
        self.model = ArmModel(2, dtype=torch.float)

    def set_rollout_model(self):
        #from robot.model.arm.acrobat.phys_model import ArmModel
        #model = ArmModel(2)
        #self.rollout_predictor = A.train.RolloutWrapper()
        #raise NotImplementedError
        super(AcrobatTrainer, self).set_rollout_model()

    def set_policy(self):
        self.set_rollout_model()
        args = self.args
        self.controller = A.train.RolloutCEM(self.rollout_predictor, self.env.action_space, iter_num=2,
                                     horizon=10, num_mutation=500, num_elite=20, device=args.device)

    def make_frame_cls(self, env_name, env):
        return AcrobatFrame


if __name__ == '__main__':
    args = A.train.get_args()
    AcrobatTrainer(args)
