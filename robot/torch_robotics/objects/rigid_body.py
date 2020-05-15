"""
Basical rigid body class, which supports query the Jacobian and the manipulation of the inertia matrices.
"""
import torch
from .physical_object import PhysicalObject
from ..arith import dot, inv_trans, Adjoint, transpose, Rp_to_trans, transform_vector, eyes_like
from .. import arith


class RigidBody(PhysicalObject):
    xdof = 9
    vdof = 6

    # the basical class to manipulate the rigid body
    # handles the rigid-body dynamics ...

    # by default we use the body frame to calcualte the qpos, qvel for ease

    def __init__(self, cmass, inertia, mass, velocity=None):
        # cmass is in SE3
        if cmass.dim() == 2:
            cmass, inertia, mass = cmass[None, :], inertia[None, :], mass[None,]
        self.cmass, self.inertia, self.mass, self.velocity = cmass, inertia, mass, velocity
        if velocity is None:
            self.velocity = self.mass.new_zeros(*self.mass.shape, 6)
        super(RigidBody, self).__init__()

    @property
    def G(self):
        out = self.inertia.new_zeros(*self.mass.shape, 6, 6)
        out[..., :3, :3] = self.inertia
        out[..., [3,4,5], [3,4,5]] = self.mass[..., None]
        return out

    def dynamics(self,  gravity=None, wrench=None):
        """
        :param wrench: the force applied from the other objects, wrench is always in the local frame..
        :param gravity: the gravity acceleration
        :return:
        """
        # get the rigid body dynamics
        # \ddot q = M^{-1}(\tau - c)=M^{-1}c, return M, c

        # Note that f+[ad_v]^TGv=Ga
        # G^{-1}(f + [ad_v]^TGv) + g = a
        G_body = self.G
        invG = torch.inverse(G_body) # we assume it's always invertible
        # TODO: we can optimize it
        adT = transpose(arith.ad(self.velocity))
        c = dot(adT, dot(G_body, self.velocity))

        if wrench is not None:
            c = c + wrench
        if gravity is not None:
            while gravity.dim() < c.dim():
                gravity = gravity[None, :]
            gravity = gravity.expand(*c.shape[:-1], -1)
            c += dot(G_body[..., 3:], dot(self.cmass[..., :3,:3].transpose(-1, -2), gravity))
        return invG, c


    def spatial_mass_matrix(self, T_a=None):
        # return the spatial mass matrix in the frame T
        # so that we can apply the newton law in the frame T

        # G_a = [Ad_{T_ba}]^T G_b [Ad_{T_ba}]
        #  T_b T_ba y = T_a y =>T_{ba} = T_b^-1 T_a

        # if T_a is None, we assume it's identity, e.g., the spatial coordinate
        T_ba = inv_trans(self.cmass)
        if T_a is not None:
            T_ba = dot(T_ba, T_a)
        Ad = Adjoint(T_ba)
        return dot(dot(transpose(Ad), self.G), Ad)

    def align_principle(self, inplace=False):
        # find the rotation to align the axis ...
        I_, v = ((self.inertia + transpose(self.inertia)) * 0.5).symeig(eigenvectors=True)
        # self.I = vI_v^T => v^Tself.Iv =I_ => R_{bc} = v
        return self.rotate(v, inplace)

    def rotate(self, R_bc, inplace=False, inertia_=None):
        # rotate about
        # I_c = R_{bc}^TI_bR_{bc}
        if inertia_ is None:
            inertia_ = dot(dot(transpose(R_bc), self.inertia), R_bc)
        cmass_ = self.cmass.clone()
        cmass_[...,:3,:3] = dot(cmass_[...,:3,:3], R_bc)
        if not inplace:
            return self.__class__(cmass_, inertia_, self.mass)
        else:
            self.cmass, self.inertia = cmass_, inertia_
            return self

    def align_principle_(self):
        return self.align_principle(inplace=True)

    def __add__(self, others):
        # return the new rigid body by summing two together
        # the two must be in the same coordinate frame
        # we support the sum of multi objects to avoid too many division
        # TODO:In fact if there are a lot of objects, we need to replace the for with the.
        if others is None:
            return self

        if isinstance(others, self.__class__):
            others = [others]

        others.append(self)
        mq, m = 0, 0
        for i in others:
            mq = i.cmass[..., :3, 3] * i.mass[..., None] + mq
            m = i.mass + m
        q = mq / m # the new position of the center of the mass, in the spatial frame
        R = self.cmass[..., :3, :3] # choose an arbitary rotation matrix
        cmass = Rp_to_trans(R, q)
        inertia_ = 0
        for i in others:
            # find the new inertia at the corresponding point

            # translate ...
            q_i = transform_vector(inv_trans(i.cmass), q)
            # I_q = I_b + m(q^TqI-qq^T)
            qtqI = (q_i ** 2).sum(dim=-1)[..., None, None] * eyes_like(cmass, 3)
            qqt = dot(q_i[..., :, None], transpose(q_i[..., :, None]))
            I_q = i.inertia + i.mass[..., None, None] * (qtqI - qqt)

            # R = R_b
            # I_c = R_{bc}^TI_bR_{bc}
            R_bc = dot(transpose(R), i.cmass[..., :3, :3])
            I_q = dot(dot(R_bc, I_q), transpose(R_bc))
            inertia_ = I_q + inertia_

        return self.__class__(cmass, inertia_, m)

    @classmethod
    def sum_boides(cls, bodies):
        mq, m = 0, 0
        for i in bodies:
            mq = i.cmass[..., :3, 3] * i.mass[..., None] + mq
            m = i.mass + m
        q = mq/m
        R = bodies[0].cmass[..., :3, :3]
        cmass = Rp_to_trans(R, q)

        G = 0
        for i in bodies:
            G = i.spatial_mass_matrix(cmass) + G # sum of the spatial inertia at the cmass
        return cls(cmass, G[..., :3, :3], m)

    def __repr__(self):
        return f"cmass: {self.cmass}\ninertia: {self.inertia}\nmass: {self.mass}\n"

    def iter(self):
        return self.cmass, self.inertia, self.mass, self.velocity

    @property
    def shape(self):
        return self.mass.shape

    def assign(self, b):
        self.cmass, self.inertia, self.mass, self.velocity = b.iter()

    def euler_(self, qacc, dt, inplace=True):
        # one-step euler integral..
        # in it's in the body frame ...
        if inplace:
            if isinstance(dt, torch.Tensor):
                dt = dt[..., None]
            self.velocity = self.velocity + qacc * dt # first update the velocity
            se3 = arith.vec_to_se3(self.velocity)
            if isinstance(dt, torch.Tensor):
                dt = dt[..., None]
            self.cmass = dot(self.cmass, arith.expse3(se3 * dt))
        else:
            raise NotImplementedError("Not inplace integral is not implemented yet")

    def kinetic(self):
        # return the current kinetic energy
        return (self.velocity * dot(self.G, self.velocity)).sum(dim=-1)/2

    def potential(self, g=9.8):
        return self.mass * self.cmass[..., 2,3] * g

    def energy(self, g=9.8):
        return self.kinetic() + self.potential(g)

    def compute_jacobian(self, pose):
        # return the jacobian for the point at the pose
        # Notice that jacobian J(q) maps the velocity from spatial velocity into the velocity in the constraint space
        # the current frame is cmass b, the destination frame is contact frame c
        # V_c = Ad_{T_{cb}}V_b
        # we assume pose and cmass are all in the space frame..
        # T_{cb} = T_sc^{-1}T_sb

        T_cb = dot(arith.inv_trans(pose), self.cmass)
        return arith.Adjoint(T_cb)

    def fk(self):
        return self.cmass
