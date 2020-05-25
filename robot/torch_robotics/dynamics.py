# module methods to handle the dynamics
import torch
import numpy as np
from .objects import RigidBody, Articulation
from . import arith as tr
from ..utils import Timer


class Mechanism:
    # used to find the parameters for the solver
    # the return is the hidden dynamic functions
    vdof = 6

    def __init__(self, rigid_body=None, articulation=None, contact_dof=3, contact_method=None):
        assert rigid_body is not None or articulation is not None
        self.rigid_body: RigidBody = rigid_body
        self.articulation: Articulation = articulation
        if self.rigid_body is not None:
            self.batch_size, self.n_obj = rigid_body.shape
        else:
            self.batch_size, self.n_obj = articulation.A.shape[0], 0
        self.contact_method = contact_method

        self.invM_obj, self.c_obj = None, None
        self.invM_art, self.c_art = None, None
        self.contact_dof = contact_dof

        self.A, self.v0, self.a0, self.d0, self.Jac = None, None, None, None, None

    def __call__(self, gravity, collision=None, tau=None, wrench=None):
        # we don't consider the wrench here
        if self.rigid_body is not None:
            self.invM_obj, self.c_obj = self.rigid_body.dynamics(gravity=gravity, wrench=wrench)
        else:
            self.invM_obj = self.c_obj = None

        if self.articulation is not None:
            self.invM_art, self.c_art = self.articulation.dynamics(gravity=gravity, tau=tau)
        else:
            self.invM_art = self.c_art = None

        if collision is not None and collision.max_nc > 0:
            self.build_contact_dynamics(collision)
        else:
            self.A = self.v0 = self.a0 = self.d0 = self.Jac = None
        return self

    def build_contact_dynamics(self, collisions):
        # some constant..
        base_object = self.invM_obj if self.rigid_body is not None else self.invM_art
        max_nc = collisions.max_nc
        batch_size = self.batch_size
        contact_dof = self.contact_dof
        device = base_object.device
        vdof = self.vdof
        dim_art = self.invM_art.shape[-1] if self.invM_art is not None else 0
        self.dimq = dimq = self.n_obj * self.vdof + dim_art

        invM = base_object.new_zeros(batch_size, dimq, dimq)
        for i in range(self.n_obj):
            invM[:, i * vdof:(i + 1) * vdof, i * vdof:(i + 1) * vdof] = self.invM_obj[:, i]
        if self.invM_art is not None:
            invM[:, -dim_art:, -dim_art:] = self.invM_art

        # --------------------------- get the jacobian matrix ----------------------------------- #
        J = invM.new_zeros(batch_size * max_nc * contact_dof * dimq)
        # object jacobian

        def assign_jac(J, batch_id, contact_id, obj_id, jac):
            obj_id = obj_id.clamp(0, self.n_obj)

            # hack
            _contact_dof_index = torch.arange(contact_dof, device=device)

            _jac = jac[:, 3:4] if contact_dof == 1 else (jac[:, 3:] if contact_dof == 3 else jac[:, [3, 4, 5, 0, 1, 2]])
            index = ((batch_id * int(max_nc) + contact_id) * contact_dof)[:, None] + _contact_dof_index[None, :]
            index = index[:, :, None] * int(dimq) + (obj_id[:, None] * int(vdof) +
                                                torch.arange(_jac.shape[-1], device=device)[None, :])[:, None]
            assert index.shape == _jac.shape
            # the 3d index... of each _jac's element
            J = J.scatter(dim=0, index=index.reshape(-1), src=_jac.reshape(-1))
            return J

        if self.invM_obj is not None:
            batch_id, contact_id, pose, obj_id, sign = collisions.filter(lambda x: (x < self.n_obj) & (x >=0))
            bodyJac = self.rigid_body[batch_id, obj_id].compute_jacobian(pose) * sign[..., None, None]
            J = assign_jac(J, batch_id, contact_id, obj_id, bodyJac)

        if self.invM_art is not None:
            batch_id, contact_id, pose, obj_id, sign = collisions.filter(lambda x: x >= self.n_obj)
            artJac = self.articulation[batch_id].compute_jacobian(obj_id-self.n_obj, pose) * sign[..., None, None]
            J = assign_jac(J, batch_id, contact_id, obj_id, artJac)
        # assign result
        self.Jac = J = J.reshape(batch_size, max_nc, contact_dof,
                      dimq).transpose(1, 2).reshape(batch_size, max_nc * contact_dof, dimq)

        JinvM = tr.dot(J, invM)
        self.A = tr.dot(JinvM, tr.transpose(J))

        velocity = []
        tau = []
        if self.rigid_body is not None:
            velocity.append(self.rigid_body.velocity.reshape(batch_size, -1))
            tau.append(self.c_obj.reshape(batch_size, -1))
        if self.articulation is not None:
            velocity.append(self.articulation.qvel)
            tau.append(self.c_art)
        self.v0 = tr.dot(J, torch.cat(velocity, dim=-1))
        self.a0 = tr.dot(JinvM, torch.cat(tau, dim=-1))

        index = collisions.batch_id * int(max_nc) + collisions.contact_id
        self.d0 = J.new_zeros(batch_size * max_nc).scatter(
            dim=0, index=index, src=collisions.dist).reshape(batch_size, max_nc)
        return


    def solve(self, dt):
        if self.A is not None:
            f = self.contact_method.solve(self, dt)
            f = tr.dot(tr.transpose(self.Jac), f)
            f = f.reshape(self.batch_size, self.dimq)
            f_obj, f_art = f[:,:self.n_obj * self.vdof], f[:,self.n_obj *self.vdof:]
            f_obj = f_obj.reshape(self.batch_size, self.n_obj, self.vdof)
        else:
            f_obj, f_art = 0, 0

        if self.invM_obj is not None:
            qacc_obj = tr.dot(self.invM_obj, self.c_obj + f_obj)
        else:
            qacc_obj = None

        if self.invM_art is not None:
            qacc_art = tr.dot(self.invM_art, self.c_art + f_art)
        else:
            qacc_art = None

        return qacc_obj, qacc_art
