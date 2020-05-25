# the code to form elastic collision for impulse based stepping system
# this is equivalent to that  n^T \cdot (v^{+}+v^{-})=0

import torch
import numpy as np
# from torch_geometric.utils import scatter_
from ..arith import dot, transpose
from ..solver import lemke
from ..solver.lcp import LCPPhysics
from ..dynamics import Mechanism


def coulomb_friction(contact_dof, A, a0, v0, d0, alpha0, mu, h, solver=None):
    # pyramidal approximation
    # we only consider 2 (n-1) along the Jacobian direction
    # the goal is

    # A : (batch, contact_dof * nc , contact_dof * nc)
    # a_0 (batch, contact_dof * nc)

    batch, nc, _ = A.shape
    nc = nc // contact_dof

    if contact_dof != 1:
        n_variable = nc * 2 * contact_dof
    else:
        n_variable = nc

    # v1 = hAf+a_0 + v0 = hA (f_1e_1 + D\beta) + a_0+v_0 = hAe_1 + hAD\beta + ha_0+v_0
    # d_1 = d_0 + v_1

    # suppose f is the nc * 2 * self.contact_dof tensor
    index = torch.arange(nc)

    e1 = A.new_zeros(batch, contact_dof * nc, n_variable)  # transform  into
    e1[:, index, index] = 1  # first nc is the  f

    if contact_dof > 1:
        D = A.new_zeros(batch, contact_dof * nc, n_variable)  # extract
        D_diagnoal = torch.arange(nc * (contact_dof-1))
        D[:, D_diagnoal + nc, D_diagnoal + nc] = 1
        D[:, D_diagnoal + nc, D_diagnoal + nc * contact_dof] = -1
    else:
        D = 0

    f = e1 + D  # use this times the variable, one can get the force

    VX, VY = h * dot(A, f), h * a0 + v0  # VX\times variable+VY is the v1

    X = A.new_zeros(batch, n_variable, n_variable)
    Y = A.new_zeros(batch, n_variable)

    # d1 -  d1_lower_bound
    X[:, :nc] = h * VX[:, :nc]
    Y[:, :nc] = h * VY[:, :nc] + d0 - alpha0

    if contact_dof > 1:
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

    sol = solver(X/h/h, Y/h/h) # /h is important for numerical issue, especially for lemke
    return dot(f, sol)


class StewartAndTrinkle:
    def __init__(self, alpha0, restitution=1., mu=0):
        self.alpha0 = alpha0
        # ideally, the restitution and mu should be the input, depends on the collision type ...
        self.restitution = restitution
        self.mu = mu
        self.solver = lemke
        #self.solver = LCPPhysics()

    def solve(self, mechanism: Mechanism, dt):
        A, a0, v0, d0, h = mechanism.A, mechanism.a0, mechanism.v0, mechanism.d0, dt
        _v0 = v0[:, :d0.shape[1]]
        d1_lower_bound = (d0 - h * _v0 * self.restitution).clamp(self.alpha0, np.inf)
        return coulomb_friction(mechanism.contact_dof, A=A, a0=a0, v0=v0, d0=d0,
                             alpha0=d1_lower_bound, mu=self.mu, h=h, solver=self.solver)
