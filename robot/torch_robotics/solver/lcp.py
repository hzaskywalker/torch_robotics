# the simplest lcp solver
# projected gauss-sediel
import numpy as np
import torch

class ProjectedGaussSiedelLCPSolver:
    def __init__(self, niters=4, max_f=np.inf):
        self.niters = niters
        self.max_f = max_f

    def __call__(self, A, a):
        """
        TODO: maybe we don't need to do it iteratively? I don't know which is better for gpu
        \dot xi = A\lambda + a; \dot xi \ge 0; \lambda\ge 0; \dot xi^T \lambda = 0 <=>
        minimize 1/2 \lambda^T A \lambda + \lambda^T a     subject to. \lambda \ge 0
        :param A: (batch, k, k)
        :param a: (batch, size)
        """
        n = A.shape[-1]
        u = torch.zeros_like(a)
        for _ in range(self.niters):
            for i in range(n):
                if i == 0:
                    u[i] = -a[:, 0]/A[:, 0, 0]
                else:
                    u[i] = -(a[:, i] + (A[:, i, :i] * u[:, :i]).sum(dim=-1))/A[:, i, i]
                u[i] = u[i].clamp(0, self.max_f)
        return u
