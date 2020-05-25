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
        # print('articulation.fk', self.qpos.shape, self.M.shape, self.A.shape)
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
        J = tr.jacobian_space(self.qpos, self.M, self.A)
        index = torch.arange(self.qpos.shape[1], device=self.M.device)[None, :].expand(link_id.shape[0], -1)
        mask = (index <= link_id[:, None]).float()
        J = J * mask[:, None]
        T_cs = tr.inv_trans(pose)
        return tr.dot(tr.Adjoint(T_cs), J)
