# the code to form elastic collision for impulse based stepping system
# this is equivalent to that  n^T \cdot (v^{+}+v^{-})=0

import torch
import numpy as np
# from torch_geometric.utils import scatter_
from ..arith import dot, transpose
from ..solver import ProjectedGaussSiedelLCPSolver
from .utils import dense_contact_dynamics

class ElasticImpulse:
    def __init__(self, alpha0):
        self.alpha0 = alpha0
        self.solver = ProjectedGaussSiedelLCPSolver()

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

        # v1 = (d1-d0)/h >= -v0 => d1 \ge -hv0 + d0
        # we also have d1 >= alpha0
        # d1 = h*h(Af+a0) + hv0 + d0
        A, a0, v0, d0, J = dense_contact_dynamics(engine, jac, invM, tau, dist, velocity)
        #print(A, a0, v0, d0, J)

        h = engine.dt
        d1_lower_bound = (d0 - h * v0).clamp(self.alpha0, np.inf)

        X = A
        Y = a0 + v0/h + (d0 - d1_lower_bound)/h/h
        f = self.solver(X, Y) #(batch, nc)
        # after we have solved the f, what shall we do for the next?
        # calculate J^Tf

        a1 = dot(transpose(J), f).transpose(1, 0).reshape(engine.batch_size * engine.n_rigid_body, invM.shape[-1])
        return a1
