# the simplest lcp solver
# projected gauss-sediel
import numpy as np
import torch
from ..arith import dot, eyes_like

class ProjectedGaussSiedelLCPSolver:
    def __init__(self, niters=5, max_f=np.inf):
        self.niters = niters
        self.max_f = max_f

    def __call__(self, A, a, niters=None):
        """
        TODO: maybe we don't need to do it iteratively? I don't know which is better for gpu
        \dot xi = A\lambda + a; \dot xi \ge 0; \lambda\ge 0; \dot xi^T \lambda = 0 <=>
        minimize 1/2 \lambda^T A \lambda + \lambda^T a     subject to. \lambda \ge 0
        :param A: (batch, k, k)
        :param a: (batch, size)
        """
        n = A.shape[-1]
        u = torch.zeros_like(a)

        if niters is None:
            niters = self.niters

        A_diag = A[:,torch.arange(n), torch.arange(n)]
        assert A_diag.min() > 1e-15, "ProjectedGaussSeidel can't solve the problem in this case"
        for _ in range(niters):
            for i in range(n):
                u = -(dot(A, u) - A_diag * u + a)/A_diag
                u = u.clamp(0, self.max_f)
        return u


class SlowLemkeAlgorithm:
    # code adapted from https://github.com/hzaskywalker/num4lcp/blob/master/matlab/lemke.m
    def __init__(self, niters=5, zer_tol=1e-5, piv_tol=1e-8):
        self.niters = niters
        self.zer_tol = zer_tol
        self.piv_tol = piv_tol

    def pivot(self, M, bas, x, entering):
        # M is a matrix of (batch, n, 2n+1), the concatenate of n and M
        # M = [M, -I, 1]
        #   M x = -q, where we require that -q is not all smaller than 0
        # bas (batch, n) is the index of active variables
        #    we require the last of bas is 2n+1
        # x  (batch, n) is the current results

        # return new x, and the new bas

        c = M.gather(dim=2, index=entering[:, None, None].expand(*M.shape[:2], 1))[:, :, 0] # (batch, n)
        B = M.gather(dim=2, index=bas[:, None, :].expand(*M.shape[:2], -1)) #(batch, n, n)

        d = torch.solve(c[:,:, None], B)[0][..., 0] # we should avoid this ...
        j = d > self.piv_tol # (batch, n) the elements which d[j] is greater than 0

        # pay attention, j should not be zero
        # x_active (batch, n)

        # theta: (batch,)
        inf = 1e9
        inv_d = 1./d.clamp(1e-15, np.inf)

        theta, _ = ((x + self.zer_tol) * inv_d + (1-j.float()) * inf).min(dim=1) # minimal x[j]+zer

        # last_one
        x_div_d = x * inv_d
        candidate = (x_div_d < theta[:, None]).float() #(batch, n)

        # weights = is_largest + random + equal_to_2n+1
        weight = candidate * d + torch.randn_like(candidate) * 1e-9 - (1-candidate) * inf
        weight[:, -1] += candidate[:, -1] * inf # for the last one, we have the largest weight

        leaving = weight.max(dim=1)[1] # choose the one to leave

        ratio = x_div_d.gather(dim=1, index=leaving[:, None])[:, 0]

        new_x = x - ratio[:, None] * d

        new_x = new_x.scatter(dim=1, index=leaving[:, None], src=ratio[:, None])
        new_bas = bas.scatter(dim=1, index=leaving[:, None], src=entering[:, None])
        return new_bas, new_x, leaving


    def evaluate(self, M, bas, x, q):
        # check the equality
        assert x.min() >= -1e-15
        B = M.gather(dim=2, index=bas[:, None, :].expand(*M.shape[:2], -1)) #(batch, n, n)
        assert ((dot(B, x) + q)**2).abs().max() < 1e-6


    def create_MI1(self, M):
        return torch.cat((M, -eyes_like(M), torch.ones_like(M[..., -1:])), dim=-1)

    def __call__(self, M, q, niters=None):
        # https://www.cs.ubc.ca/cgi-bin/tr/2005/TR-2005-01.pdf
        n = M.shape[-1]
        if niters is not None:
            self.niters = niters
        for i in range(self.n * self.niters):
            pass
