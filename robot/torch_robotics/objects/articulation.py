# articulation with Newton Euler Method
# currently we only support the very simple one
import torch
import numpy as np
from .physical_object import PhysicalObject
from .. import arith as tr
from robot.utils import batched_index_select


class Articulation(PhysicalObject):
    # description of the articulation
    # Articulation is in general in the form form of (batch_size, n_links+1)
    # each link and the end effector is regarded as a rigid_body
    mod = np.pi * 2
    def __init__(self, qpos, qvel, M, A, G):
        assert len(qpos.shape) == 2
        self.M = M
        self.A = A
        self.G = G
        self.qpos = qpos
        self.qvel = qvel
        super(Articulation, self).__init__()

    def iter(self):
        return (self.qpos, self.qvel, self.M, self.A, self.G)

    def fk(self):
        # return the pose that can be used for the render and the articulation
        return tr.fk_in_space(self.qpos, self.M, self.A)

    def euler_(self, qacc, dt, inplace=True):
        assert inplace
        # in place integration
        self.qvel = self.qvel + qacc * dt  # first update the velocity
        self.qpos = self.qpos + self.qvel * dt
        if self.mod is not None:
            self.qpos = (self.qpos + self.mod/2) % self.mod - self.mod/2

    def dynamics(self, gravity, tau=None, ftip=None):
        if tau is None:
            tau = torch.zeros_like(self.qpos)
        if ftip is None:
            ftip = tau.new_zeros(*tau.shape[:-1], 6)
        else:
            raise NotImplementedError
        while gravity.dim() < tau.dim():
            gravity = gravity[None,:]
        gravity = gravity.expand(*tau.shape[:-1], -1)
        mass, c, g, f = tr.compute_all_dynamic_parameters(self.qpos, self.qvel, gravity, ftip,
                                                          self.M, self.G, self.A)
        return torch.inverse(mass), tau-c-g-f

    def compute_jacobian(self, link_id, pose):
        # T(q) = (\prod e^{[S_i\theta_i]})M
        S = tr.A_to_S(self.A, self.M) # spatial coordinate
        Js = tr.transpose(S.clone())
        Js[..., 1:] = 0
        T = tr.eyes_like(self.M[:, 0, :, :], 4)

        outs, Ts = [Js], []
        for i in range(1, self.qpos.shape[-1]+1):
            T = tr.dot(T, tr.expse3(tr.vec_to_se3(S[:, i - 1]) * self.qpos[:, i - 1][:, None, None]))
            Ts.append(T)

            if i != self.qpos.shape[-1]:
                Js[:, :, i] = tr.dot(tr.Adjoint(T), S[:, i])
            else:
                Js = tr.dot(tr.Adjoint(self.M[..., -1]), Js)
            outs.append(Js)

        Ts.append(tr.dot(T, self.M[..., -1, :, :]))

        outs = torch.stack(outs, dim=1)
        Ts = torch.stack(Ts, dim=1)

        J = batched_index_select(outs, 1, link_id)[:, 0]
        T = batched_index_select(Ts, 1, link_id)[:, 0]

        # Ad_{T_{link, pose}} J\dot q =
        T_cb = tr.dot(tr.inv_trans(pose), T)
        return tr.dot(tr.Adjoint(T_cb), J)
