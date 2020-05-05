# the code to form elastic collision for impulse based stepping system
# this is equivalent to that  n^T \cdot (v^{+}+v^{-})=0

import torch
import numpy as np
# from torch_geometric.utils import scatter_
from ..arith import dot, transpose, eyes_like
from ..solver import ProjectedGaussSiedelLCPSolver, lemke
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
        sol = solver(X/h/h, Y/h/h) # /h is important for numerical issue, especially for lemke
        f = dot(f, sol)
        return f
    else:
        return X, Y, VX, VY, f


class ElasticImpulse:
    def __init__(self, alpha0, restitution=1., contact_dof=1, mu=0, use_toi=True):
        self.alpha0 = alpha0
        self.contact_dof = contact_dof

        # ideally, the restitution and mu should be the input, depends on the collision type ...
        self.restitution = restitution
        self.mu = mu
        #if contact_dof == 1:
        #    self.solver = ProjectedGaussSiedelLCPSolver()
        #else:

        # we always use the lemke solver as we find it's faster than PGS for linear case...
        self.solver = lemke
        self.use_toi = use_toi

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

        h = engine.dt
        _v0 = v0

        if self.contact_dof == 1:
            if self.use_toi:
                toi = (-d0 / (v0 + (v0.abs() < 1e-15).float())).clamp(0, h)
            else:
                toi = d0 * 0
            #X = A * ((h-toi) * (h-toi))[:, None]
            #Y = a0 * h * h + v0 * h + d0

            # we consider the following approximation
            #    d_1 = d_0 + h^2 a_0 + hv_0 + h(h-toi) Af
            #    v_1 = ha_0 + v_0 + (h-toi)Af = (d_1 - d_0)/h >= -r * v_0 => d_1 >= d_0 - r * h * v_0
            t2 = h - toi
            X = A * h * t2[:, :, None]

            Y = a0 * h * h + v0 * h + d0

            elastic = d0  - h * self.restitution * _v0  # inverse the velocity
            d1_lower_bound = elastic.clamp(self.alpha0, np.inf)
            Y = Y - d1_lower_bound

            f = self.solver(X/h/h, Y/h/h) /h * t2 #(batch, nc)
        else:
            _v0 = _v0[:, :d0.shape[1]]
            d1_lower_bound = (d0 - h * _v0).clamp(self.alpha0, np.inf)
            f = coulomb_friction(self.contact_dof, A=A, a0=a0, v0=v0, d0=d0,
                                 alpha0=d1_lower_bound, mu=self.mu, h=h, solver=self.solver)

        f = dot(transpose(J), f)
        # f batch, dimq = f, batch, nobj, vbof
        f = f.reshape(engine.batch_size, engine.n_rigid_body,
                      invM.shape[-1]).transpose(0, 1).reshape(-1, invM.shape[-1])
        a1 = dot(invM, f)
        return a1
