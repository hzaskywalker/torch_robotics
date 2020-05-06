# the simplest lcp solver
# projected gauss-sediel
# http://image.diku.dk/kenny/download/erleben.13.siggraph.course.notes.pdf
import numpy as np
import torch
from ..arith import dot, eyes_like

def bmv(m, v):
    # batched matrix vector multiplication
    return (m@v[..., None])[..., 0]


def backward(M, q, u, grad_u):
    # TODO: may be not correct ...
    from torch.nn.functional import relu
    # u is the solution to Mu+q\ge 0, u\ge 0, (Mu+q)\cdot u = 0
    assert M.dim() == 3 and q.dim() == 2 and u.dim() == 2
    batch_size, n = u.shape
    with torch.no_grad():
        eye = torch.eye(n, device=M.device, dtype=M.dtype)
        u = relu(u)
        xi = relu(bmv(M, u) + q)

        P = M.new_zeros(batch_size, 2*n, 2*n)
        A = M
        B = eye
        C = eye[None,:] * xi[:, None]
        D = eye[None,:] * u[:, None]
        P[:, :n, :n] = A
        P[:, :n, n:] = B
        P[:, n:, :n] = C
        P[:, n:, n:] = D

        to_inv = (grad_u.abs() > 1e-15).any(dim=-1)
        d_u = torch.zeros_like(grad_u)

        #print(P[to_inv][5], grad_u[to_inv][5], u[to_inv][0])
        #print(P[to_inv][0])
        P = P[to_inv]
        P = P + eyes_like(P) * 1e-10 # TODO: small hack to make it not singular ...
        d_u[to_inv] = -bmv(torch.inverse(P)[:, :n, :n], grad_u[to_inv])
        return u[:, None, :] * d_u[:, :, None], d_u

def backward2(M, q, u, grad_u):
    from torch.nn.functional import relu
    # u is the solution to Mu+q\ge 0, u\ge 0, (Mu+q)\cdot u = 0
    # we consider the gradient to the following problem
    # 1/2 z^TMz+ q^Tz st. Gz<=h where G=[-M\\ -I], h=[q\\0]

    assert M.dim() == 3 and q.dim() == 2 and u.dim() == 2
    batch_size, n = u.shape
    with torch.no_grad():
        eye = torch.eye(n, device=M.device, dtype=M.dtype)
        eye2 = torch.eye(2*n, device=M.device, dtype=M.dtype)
        u = relu(u)
        xi = relu(bmv(M, u) + q)

        P = M.new_zeros(batch_size, n*3, n*3)
        lamb = M.new_zeros(batch_size, n*2)
        lamb[:, n:] = xi
        """
        given lambda*
        P is 
        Q G^T
        D(\lambda*)G D(Gz^*-h)
        """
        G = torch.cat((-M, -eye[None,:].expand(batch_size, -1, -1)), dim=1)
        h = torch.cat((q, torch.zeros_like(q)), dim=1)
        A = M
        B = G.transpose(1, 2)
        C = dot(eye2[None, :] * lamb[:, None], G)
        D = eye2[None, :] * (-relu(-bmv(G, u) + h)[:, None])
        P[:, :n, :n] = A
        P[:, :n, n:] = B
        P[:, n:, :n] = C
        P[:, n:, n:] = D

        to_inv = (grad_u.abs() > 1e-15).any(dim=-1)
        d_u = torch.zeros_like(grad_u)

        #print(P[to_inv][5], grad_u[to_inv][5], u[to_inv][0])
        #print(P[to_inv][0])
        P = P[to_inv]
        P = P + eyes_like(P) * 1e-10 # TODO: small hack to make it not singular ...
        d_u[to_inv] = -bmv(torch.inverse(P.transpose(-1, -2))[:, :n, :n], grad_u[to_inv])
        return u[:, None, :] * d_u[:, :, None], d_u


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
        assert A_diag.min() > 1e-8, "ProjectedGaussSeidel can't solve the problem in this case"
        for _ in range(max(niters, n//2)):
            for i in range(n):
                new_u = -((A[:, i] * u).sum(dim=-1) + a[:, i])/A_diag[:, i] + u[:, i]
                u = u.clone()
                u[:, i] = u[:, i] + 1. * (new_u - u[:, i])
                u = u.clamp(0, self.max_f)
        return u


class SlowLemkeAlgorithm:
    # code adapted from https://github.com/hzaskywalker/num4lcp/blob/master/matlab/lemke.m
    def __init__(self, niters=1000, zer_tol=1e-5, piv_tol=1e-8):
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

        inf = 1e9
        # this is important...
        inv_d = 1./d.clamp(self.piv_tol, np.inf)

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
        return new_bas, new_x, bas.gather(dim=1, index=leaving[:, None])[:, 0]

    def reverse(self, bas, entering):
        n = bas.shape[-1]
        entering = (entering-n).clamp(0, n) + (entering+n)*(entering<n).long()
        return entering

    def solve(self, M, bas, xs, entering, num_iter=100):
        """
        print('entering', entering)
        print('M', M)
        print('bas', bas)
        print('xs', xs)
        """
        new_bas, new_xs, leaving = self.pivot(M, bas, xs, entering)
        last = M.shape[-1]
        unsolved = (leaving != last - 1)
        answer = new_xs.new_zeros((M.shape[0], M.shape[-1]))
        answer.scatter_(dim=1, index=new_bas, src=new_xs)
        if not num_iter:
            import logging
            logging.warning("WRONG>>>>>>>>>>>>>>>>>>>>>>>>>>> LEMKE doesn't converge after 1000 timesteps")
            return None
        if unsolved.sum() > 0 and num_iter>0:
            rx = self.solve(M[unsolved], new_bas[unsolved],
                           new_xs[unsolved], self.reverse(bas, leaving[unsolved]), num_iter-1)
            if rx is not None:
                answer[unsolved] = rx
        return answer

    def evaluate(self, M, bas, x, q):
        # check the equality
        assert x.min() >= -1e-15
        B = M.gather(dim=2, index=bas[:, None, :].expand(*M.shape[:2], -1)) #(batch, n, n)
        assert ((dot(B, x) + q)**2).abs().max() < 1e-6

    def create_MI1(self, M):
        return torch.cat((M, -eyes_like(M), torch.ones_like(M[..., -1:])), dim=-1)

    def init(self, M, q):
        # Mx = -q
        def swap_and_set_last(x, index, val):
            # set the index to be the last row
            # we want to keep the last index reamaining at the last..
            x = x.scatter(dim=-1, index=index[:, None], src=x[:, -1:])
            x[:, -1] = val
            return x

        n = M.shape[-1]
        z0, index = q.min(dim=-1)
        z0 = -z0 # z0 is the inverse of minimal q, we assume there is always a i, q_i<0
        x0 = (q + z0[..., None])
        bas = (n+torch.arange(n, device=M.device))[None,:].expand(M.shape[0], -1)

        x0 = swap_and_set_last(x0, index, z0)
        bas = swap_and_set_last(bas, index, 2*n)
        return bas, x0, index

    def run(self, M, q, niters):
        n = M.shape[-1]
        if niters is None:
            niters = self.niters

        init_bas, init_x, init_entering = self.init(M, q)
        M = self.create_MI1(M)
        answer = self.solve(M, init_bas, init_x, init_entering, num_iter=niters)
        return answer[..., :n]


    def __call__(self, M, q, niters=None):
        # https://www.cs.ubc.ca/cgi-bin/tr/2005/TR-2005-01.pdf
        unsolved = (q<0).any(dim=-1) # must be smaller than 1e-3, otherwise lemke will not terminate
        answer = M.new_zeros((M.shape[0], M.shape[-1]))

        if unsolved.any():
            out = self.run(M[unsolved], q[unsolved], niters)
            if out is not None:
                answer[unsolved] = out

        #TODO: we can check if the solution to the LCP is correct
        #self.check(M, q, answer)
        return answer

    def check(self, M, q, x):
        assert ((dot(M, x) + q) > -self.zer_tol).all(), f"{(dot(M, x)+q).min()}"
        assert (((dot(M, x) + q) * x).abs() < self.piv_tol).all(), f"{((dot(M, x) + q) * x).abs().max()}"


forward_alg = SlowLemkeAlgorithm(2000, 1e-5, 1e-8)


from torch.autograd import Function
class FasterSlowLemke(Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, M, q, niters=None):
        with torch.no_grad():
            u = forward_alg(M, q, niters)
        ctx.save_for_backward(M, q, u)
        return u

    @staticmethod
    def backward(ctx, grad_output):
        M, q, u = ctx.saved_tensors
        M_grad, q_grad = backward(M, q, u, grad_output)
        return M_grad, q_grad, None
lemke = FasterSlowLemke.apply

class FasterSlowLemke2(Function):
    @staticmethod
    def backward(ctx, grad_output):
        M, q, u = ctx.saved_tensors
        M_grad, q_grad = backward2(M, q, u, grad_output)
        return M_grad, q_grad, None
lemke2 = FasterSlowLemke2.apply

class CvxpySolver:
    # NOTE: we found the default parameter is not very accurate ...
    #  perhaps we should try the ipm ..
    def __init__(self, n):
        import cvxpy as cp
        from cvxpylayers.torch import CvxpyLayer
        x = cp.Variable(n)
        A = cp.Parameter((n, n))
        b = cp.Parameter(n)
        objective = cp.Minimize(0.5 * cp.sum_squares(A@x) + x.T @ b)
        constraints = [x >= 0]
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()
        cvxpylayer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
        self.cvxpylayer = cvxpylayer

    def __call__(self, M, q):
        return self.cvxpylayer(torch.cholesky(M).transpose(-1, -2), q)[0]


class QpthSolver:
    def __init__(self, max_iter=20):
        from qpth.qp import QPFunction
        self.f = QPFunction(eps=1e-12, verbose=True, maxIter=max_iter, notImprovedLim=10)

    def __call__(self, M, q):
        A = torch.tensor([], dtype=M.dtype, device=M.device)
        b = torch.tensor([], dtype=M.dtype, device=M.device)
        G = M.new_zeros(M.shape[0], M.shape[1] * 2, M.shape[2])
        G[:,:M.shape[1]] = -M
        G[:,M.shape[1]:] = -eyes_like(M)
        h = M.new_zeros(M.shape[0], M.shape[1] * 2)
        h[:,:M.shape[1]] = q
        return self.f(M, q, G, h, A, b)


class LCPPhysics:
    def __init__(self):
        from lcp_physics.lcp.lcp import LCPFunction
        self.f = LCPFunction(max_iter=30, verbose=True, not_improved_lim=5, eps=1e-15)

    def __call__(self, M, q):
        n = M.shape[-1]
        # Q, p, G, h, A, b, F
        # M  q  G  m        -F

        # in their code they requre sz=0
        A = torch.tensor([])
        b = torch.tensor([])
        h = q
        Q = eyes_like(M)
        p = q * 0
        G = -eyes_like(M)
        F = M + G
        out = self.f(Q, p, G, h, A, b, F)
        return out
