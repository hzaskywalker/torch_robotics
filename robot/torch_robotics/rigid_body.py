"""
Basical rigid body class, which supports query the Jacobian and the manipulation of the inertia matrices.
"""
from .base import dot, inv_trans, Adjoint, transpose, Rp_to_trans, transform_vector, eyes_like

#-----------------------  Rigid Body -------------------
class RigidBody:
    # the basical class to manipulate the rigid body
    # handles the rigid-body dynamics ...

    def __init__(self, cmass, inertia, mass):
        # cmass is in SE3
        if cmass.dim() == 2:
            cmass, inertia, mass = cmass[None,:], inertia[None,:], mass[None,]
        self.cmass, self.inertia, self.mass = cmass, inertia, mass

    @property
    def G(self):
        out = self.inertia.new_zeros(*self.mass.shape, 6, 6)
        out[..., :3,:3] = self.inertia
        out[..., [3,4,5],[3,4,5]] = self.mass
        return out

    def spatial_mass_matrix(self, T_a):
        # return the spatial mass matrix in the frame T
        # so that we can apply the newton law in the frame T

        # G_a = [Ad_{T_ba}]^T G_b [Ad_{T_ba}]
        #  T_b T_ba y = T_a y =>T_{ba} = T_b^-1 T_a
        T_ba = dot(inv_trans(self.cmass), T_a)
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

    def __getitem__(self, item):
        return self.__class__(self.cmass[item], self.inertia[item], self.mass[item])

    @property
    def shape(self):
        return self.mass.shape



