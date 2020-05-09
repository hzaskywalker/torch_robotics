# the naive version of dual mujoco's formulation
# instead of use PGS to solve the cone constrained quadratic programming, we transfrom the quadratic programming
#   into the cone constrained LCP, similar to Stewart & Trinkle's approach
# and we can perhaps try to solve it with lemke?
from .elastic import StewartAndTrinkle
from ..solver import lemke
from .utils import dense_contact_dynamics
from .. import arith
import torch

class DualMujoco(StewartAndTrinkle):
    def __init__(self, alpha0=0.001, dmin=0.9, dmax=0.95, width=0.001, damping=1, stiffness=1,
                 contact_dof=1, mu=0, mid=0.5, power=2):
        assert power == 2 and mid == 0.5
        self.solver = lemke
        self.alpha0 = alpha0

        self.dmax = dmax
        self.dmin = dmin
        self.width = width

        self.b = damping/self.dmax
        self.k = stiffness/self.dmax/self.dmax

        self.solver = lemke
        self.contact_dof = contact_dof
        self.mu = mu

        assert self.contact_dof == 1


    def d(self, r):
        return torch.sigmoid(torch.exp((r/self.width)**2)) * (self.dmax - self.dmin) + self.dmin

    def __call__(self, engine, sparse_jac, invM, tau, dist, velocity):
        A, a0, v0, d0, J = dense_contact_dynamics(engine, sparse_jac,invM, tau, dist,
                                                  velocity, contact_dof=self.contact_dof)


        # the target velocity
        v_star = - self.b * v0 - self.k * d0

        # ideally this should be a parameter of the network...
        r = torch.relu(self.alpha0 - d0) # positive residual, it > 0 if and only if d0 < self.alpha0
        d_i = self.d(r) #d0 is the r for the normal direction
        epsilon = (-d_i + 1)/d_i

        R = arith.dot(arith.eyes_like(A) * epsilon[..., None, :], A)

        # the goal is to find the f in K such that
        # f = argmin 1/2f^T(A+R)f + lambda^T (a^0 - a^*)
        X = A + R
        h = engine.dt
        Y = (v0 + a0 * h - v_star)/h
        print(arith.tocpu(v_star)[:10])
        f = self.solver(X, Y)
        return self.f2a(engine, J, f, invM)

