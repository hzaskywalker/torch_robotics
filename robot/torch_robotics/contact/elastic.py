# the code to form elastic collision for impulse based stepping system
# this is equivalent to that  n^T \cdot (v^{+}+v^{-})=0

import torch
import numpy as np
# from torch_geometric.utils import scatter_
from ..arith import dot, transpose, eyes_like
from ..solver import ProjectedGaussSiedelLCPSolver, SlowLemkeAlgorithm
from .utils import dense_contact_dynamics


def coulomb_friction(contact_dof, A, a0, v0, d0, alpha0, mu, h, solver=None):
    # pyramidal approximation
    # we only consider 2 (n-1) along the Jacobian direction
    # the goal is

    # A : (batch, contact_dof * nc , contact_dof * nc)
    # a_0 (batch, contact_dof * nc)

    batch, nc, _ = A.shape
    nc = nc // contact_dof

    n_variable = nc * 2 * contact_dof

    # v1 = hAf+a_0 + v0 = hA (f_1e_1 + D\beta) + a_0+v_0 = hAe_1 + hAD\beta + ha_0+v_0
    # d_1 = d_0 + v_1

    # suppose f is the nc * 2 * self.contact_dof tensor
    index = torch.arange(nc)

    e1 = A.new_zeros(batch, contact_dof * nc, n_variable)  # transform  into
    e1[:, index, index] = 1  # first nc is the  f

    D = A.new_zeros(batch, contact_dof * nc, n_variable)  # extract
    D_diagnoal = torch.arange(nc * (contact_dof-1))
    D[:, D_diagnoal + nc, D_diagnoal + nc] = 1
    D[:, D_diagnoal + nc, D_diagnoal + nc * contact_dof] = -1

    f = e1 + D  # use this times the variable, one can get the force

    VX, VY = h * dot(A, f), h * a0 + v0  # VX\times variable+VY is the v1

    X = A.new_zeros(batch, n_variable, n_variable)
    Y = A.new_zeros(batch, n_variable)

    # d1 -  d1_lower_bound
    X[:, :nc] = h * VX[:, :nc]
    Y[:, :nc] = h * VY[:, :nc] + d0 - alpha0

    eye_nc = torch.eye(nc, device=X.device)

    # lambda e + D^Tv = 11111 + D^TVx, D^Tvy
    beta_slice = slice(nc, nc + nc * (contact_dof - 1) * 2)
    DT = transpose(D)[:, beta_slice]
    X[:, beta_slice] = dot(DT, VX)
    X[:, beta_slice, -nc:] = eye_nc.repeat(contact_dof * 2 - 2, 1) # \lambad
    Y[:, beta_slice] = dot(DT, VY)

    # mu f1 - e^T\beta
    X[:, -nc:] = mu * e1[:, :nc]  # extract f1
    X[:, -nc:, nc:-nc] = -eye_nc[:, None, :].repeat(1, contact_dof * 2 - 2, 1).reshape(nc, -1)

    if solver is not None:
        sol = solver(X, Y)
        f = dot(f, sol)
        return f
    else:
        return X, Y, VX, VY, f


class ElasticImpulse:
    def __init__(self, alpha0, restitution=1., contact_dof=1, mu=0):
        self.alpha0 = alpha0
        self.contact_dof = contact_dof

        # ideally, the restitution and mu should be the input, depends on the collision type ...
        self.restitution = restitution
        self.mu = mu
        if contact_dof == 1:
            self.solver = ProjectedGaussSiedelLCPSolver()
        else:
            self.solver = SlowLemkeAlgorithm()

    def __call__(self, engine, jac, invM, tau, dist, velocity):
        # we now have the following matrix
        #   Af+a0=a
        #   (v1-v0) = A hf + ha0
        #   d1 = d0 + h v1

        # Force: f should be greater than zero
        # LCP: f\dot (d-\alpha_0) = 0, e.g.,
        #   if f\neq 0, they must contact with each other
        #   if d>\alpha_0, then there should be no force
        # Separation: d1 \ge \alpha_0
        # Elastic collision: v1\ge -v_0; after the collision, the velocity should reverse the sign

        A, a0, v0, d0, J = dense_contact_dynamics(engine, jac, invM, tau, dist, velocity, contact_dof=self.contact_dof)
        #print(A, a0, v0, d0, J)

        h = engine.dt

        _v0 = v0
        if self.contact_dof > 1:
            _v0 = _v0[:, :d0.shape[1]]
        d1_lower_bound = (d0 - h * _v0).clamp(self.alpha0, np.inf)

        if self.contact_dof == 1:
            # we don't consider friction here
            # v1 = (d1-d0)/h >= -v0 => d1 \ge -hv0 + d0
            # we also have d1 >= alpha0
            # d1 = h*h(Af+a0) + hv0 + d0


            X = A
            Y = a0 + v0/h + (d0 - d1_lower_bound)/h/h
            f = self.solver(X, Y) #(batch, nc)
            # after we have solved the f, what shall we do for the next?
            # calculate J^Tf
        else:
            f = coulomb_friction(self.contact_dof, A=A, a0=a0, v0=v0, d0=d0,
                                 alpha0=d1_lower_bound, mu=self.mu, h=h, solver=self.solver)

        a1 = dot(invM, dot(transpose(J), f).transpose(1, 0).reshape(
            engine.batch_size * engine.n_rigid_body, invM.shape[-1]))

        return a1
