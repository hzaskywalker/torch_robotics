# module methods to handle the dynamics
import torch
import numpy as np
from .objects import RigidBody, Articulation
from . import arith as tr


class Collision:
    def __init__(self, shapes, collision_checker):
        # each shape has an object id
        self.shapes = shapes
        self.collision_checker = collision_checker

        self.max_nc = 0
        # in the form of (num_contacts,)
        self.batch_id = []
        self.contact_id = []
        self.dist = []
        self.pose = []

        # list of object_id and contact_id
        # (num_contact, 2)
        # represent the two object id of contact
        # num_of_objects + num of links in articulation
        self.contact_objects = []

    def update(self):
        values = list(self.shapes.values())
        assert len(values) > 0
        batch_size = values[0].pose.shape[0]

        # count the number of contacts per batch
        batch_nc = np.zeros((batch_size,), dtype=np.int32) # contact id in batch
        self.batch_id, self.contact_id, self.dist, self.pose, self.contact_objects = [], [], [], [], []

        for i in range(len(values)):
            a_id = values[i].object_id
            for j in range(i+1, len(values)):
                b_id = values[i].object_id

                batch_id, dists, poses = self.collision_checker(values[i], values[j])

                if batch_id.shape[0] > 0:
                    # number of contacts
                    _batch_id = batch_id.detach().cpu().numpy()
                    _contact_id = np.zeros(len(batch_id))
                    for b in _batch_id:
                        batch_nc[b] += 1

                    # currently a tensor
                    self.batch_id.append(batch_id)
                    self.dist.append(dists)
                    self.pose.append(poses)

                    # current a np array
                    self.contact_id.append(_contact_id)
                    self.contact_objects.append(np.array([a_id, b_id])[None,:].repeat(len(batch_id), 0))

        self.max_nc = 0
        if len(self.batch_id)>0:
            self.max_nc = batch_nc.max()
            self.batch_id = torch.cat(self.batch_id, dim=0)
            self.dist = torch.cat(self.dist, dim=0)
            self.pose = torch.cat(self.pose, dim=0)
            device = self.batch_id.device
            self.contact_id = torch.tensor(np.concatenate(self.contact_id), dtype=torch.long, device=device)
            self.contact_objects = torch.tensor(np.concatenate(self.contact_objects), dtype=torch.long, device=device)

    def filter(self, fn):
        object_mask = fn(self.contact_objects)
        batch_id, contact_id, dist, pose, obj_id = [], [], [], [], []
        for i in range(2):
            mask = object_mask[:, 0]
            if mask.any():
                batch_id.append(self.batch_id[mask])
                contact_id.append(self.contact_id[mask])
                dist.append(self.dist[mask])
                pose.append(self.pose[mask])
                obj_id.append(self.contact_objects[:, i][mask])
        return torch.cat(batch_id), torch.cat(contact_id), torch.cat(pose), torch.cat(obj_id)


class DynamicsEquation:
    # used to find the parameters for the solver
    # the return is the hidden dynamic functions
    vdof = 6

    def __init__(self, rigid_body, articulation=None, contact_dof=3):
        assert self.rigid_body is not None
        self.rigid_body = rigid_body
        self.articulation = articulation
        self.batch_size, self.n_obj = self.rigid_body.shape

        self.invM_obj, self.c_obj = None, None
        self.invM_art, self.c_art = None, None
        self.contact_dof = contact_dof

        self.A, self.v0, self.a0, self.d0, self.Jac = None, None, None, None, None

    def __call__(self, gravity, collision: Collision=None, tau=None, wrench=None):
        # we don't consider the wrench here
        if self.rigid_body is not None:
            self.invM_obj, self.c_obj = self.rigid_body.dynamics(gravity=gravity, wrench=wrench)
        else:
            self.invM_obj = self.c_obj = None

        if self.articulation is not None:
            self.invM_art, self.c_art = self.articulation.dynamics(gravity=gravity, tau=tau)
        else:
            self.invM_art = self.c_art = None

        if collision is not None:
            self.build_contact_dynamics(collision)
        else:
            self.A = self.v0 = self.a0 = self.d0 = self.Jac = None

    def build_contact_dynamics(self, collisions: Collision):
        # some constant..
        max_nc = collisions.max_nc
        batch_size = self.batch_size
        contact_dof = self.contact_dof
        device = self.invM_obj.device
        vdof = self.vdof
        dim_art = self.invM_art.shape[-1] if self.invM_art is not None else 0
        dimq = self.n_obj * self.vdof + dim_art

        invM = self.invM_obj.new_zeros(batch_size, dimq, dimq)
        for i in range(self.n_obj):
            invM[:, i * vdof:(i + 1) * vdof, i * vdof:(i + 1) * vdof] = self.invM_obj[:, i]
        if self.invM_art is not None:
            invM[:, -dim_art:, -dim_art:] = self.invM_art


        # --------------------------- get the jacobian matrix ----------------------------------- #
        J = invM.new_zeros(batch_size * max_nc * contact_dof * dimq)
        # object jacobian
        _contact_dof_index = torch.arange(contact_dof, device=device)

        def assign_jac(J, batch_id, contact_id, obj_id, jac):
            obj_id = obj_id.clamp(0, self.n_obj)
            _jac = jac[:, 3:4] if contact_dof == 1 else (jac[:, 3:] if contact_dof == 3 else jac[:, [3, 0, 1, 2, 4, 5]])
            # Jac (0:contact_dof, l:l+p)
            index = ((batch_id * max_nc + contact_id) * contact_dof)[:, None] + _contact_dof_index[None, :]
            index = index[:, :, None] * dimq + (obj_id[:, None] * vdof +
                                                torch.arange(_jac.shape[-1], device=device)[None, :])[:, None]
            # the 3d index... of each _jac's element
            J = J.scatter(dim=0, index=index.reshape(-1), src=_jac.reshape(-1))
            return J

        if self.invM_obj is not None:
            batch_id, contact_id, pose, obj_id = collisions.filter(lambda x: x < self.n_obj)
            bodyJac = self.rigid_body[batch_id, obj_id].compute_jacobian(pose)
            J = assign_jac(J, batch_id, contact_id, obj_id, bodyJac)

        if self.invM_art is not None:
            batch_id, contact_id, pose, obj_id = collisions.filter(lambda x: x >= self.n_obj)
            artJac = self.articulation[batch_id].compute_jacobian(obj_id-self.n_obj, pose) # get the link id
            J = assign_jac(J, batch_id, contact_id, obj_id, artJac)

        # assign result
        self.Jac = J = J.reshape(batch_size, max_nc, contact_dof,
                      dimq).transpose(1, 2).reshape(batch_size, max_nc * contact_dof, dimq)

        JinvM = tr.dot(J, invM)
        self.A = tr.dot(JinvM, tr.transpose(J))

        velocity = self.rigid_body.velocity.reshape(batch_size, -1)
        tau = self.c_obj.reshape(batch_size, -1)
        if self.articulation is not None:
            velocity = torch.cat((velocity, self.articulation.qvel), dim=-1)
            tau = torch.cat((tau, self.c_art), dim=-1)
        self.v0 = tr.dot(J, velocity)
        self.a0 = tr.dot(JinvM, tau)

        index = collisions.batch_id * max_nc + collisions.contact_id
        self.d0 = J.new_zeros(batch_size * max_nc).scatter(
            dim=0, index=index, src=collisions.dist).reshape(batch_size, max_nc)
