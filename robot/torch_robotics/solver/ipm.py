# slow version of PDIPM with second order cone
import numpy as np
import torch
from .cone import Orthant, SecondOrder, CartesianCone
from ..arith import dot, transpose

class Solver:
    # coneqp, without equality constraints ...
    # one can always eliminate teh equality constraints by simple mathematics

    def __init__(self, MAXITERS=100, feastol=1e-7, abstol=1e-7, reltol=1e-6, STEP=0.99, EXPON=3):
        self.MAXITERS = MAXITERS
        self.feastol = feastol
        self.abstol = abstol
        self.reltol = reltol
        self.STEP = STEP
        self.EXPON = EXPON

        self.cone = None

    def factor_kkt(self, P, q, G, h):
        # currently we don't do the factor
        self.P, self.q, self.G, self.h = P, q, G, h

        self.resx0 = self.q.norm(dim=-1).clamp(1.0, np.inf)
        self.resz0 = self.cone.snrm2(h).clamp(1.0, np.inf)

        self.GT = GT = G.transpose(-1, -2)

        batch_size, n, m = P.shape[0], P.shape[1], G.shape[1]
        matrix = G.new_zeros((batch_size, n+m, n+m))
        matrix[:, :n, :n] = P
        matrix[:, :n, n:] = GT
        matrix[:, n:, :n] = G
        self.n = n
        self.m = m
        self.w_slice = slice(n, None, None)
        self.matrix = matrix

    def solve_kkt(self, W, bx, bz):
        #  [ P  G'  ] [ ux ]   [ bx ]
        #  [ G  -W'W] [ uz ]   [ bz ]
        # TODO: we can avoid one matrix inverse by LU factorizing the
        M = self.matrix.clone()
        M[:, self.w_slice, self.w_slice] = - dot(W, transpose(W))
        B = torch.cat((bx, bz), dim=-1)
        out = torch.solve(B[:, :, None], M)[0][:, :, 0]
        return out[:, :self.n], out[:, self.n:]

    def initialize(self):
        # this is the f3 in coneqp
        # we initialize with the solution to 1/2 x^TPx + c^Tx + (1/2)||s||^2_2, s.t. Gx + s = h
        # solve KKT (Gz-h=z => Px+G'z +c = Px + c + G'Gx=0
        #  [ P  G'  ] [ x ]   [ -c ]
        #  [ G  -I  ] [ z ]   [ h  ]
        cone =self.cone
        x, z = self.solve_kkt(torch.eye(self.m, device=self.matrix.device), -self.q, self.h)

        # h-Gx = -z is the residual, alpha_p = inf {\alpha|-z+\alpha e\ge 0}
        # if alpha_p < 0, s = -z, otherwise -z + (1+\alpha_p) e

        def project(s):
            nrms = cone.snrm2(s)
            ts = cone.max_step(s)
            cond = ts >= -1e-8 * nrms.clamp(1.0, np.inf)
            return s + cond.float() * (1+ts) * cone.identity[None, :]

        s = project(-z) # make s >= 0 in the cone
        z = project(z)
        return x, s, z

    def Newton(self, W, dx, dz, ds, lmbda):
        # solve the following newton equation
        #  [0 ] + [ P  G' ] [ Dx ] = [ dx ]
        #  [Ds] + [ G  0  ] [ Dz ]   [ dz ]
        #  lambda o (Wdz + (W^{-T})ds) = ds
        # we can transform it into
        #  [ P  G'    ] [ Dx ] = [ dx ]
        #  [ G  -W'W  ] [ Dz ]   [ dz - W'(lambda <> ds) ]
        cone = self.cone
        tmp = cone.scale(W, cone.sinv(lmbda, ds), trans=True)

        W_mat = cone.as_matrix(W)
        #print('s', ds, 'lmbda', lmbda)
        #print('sinv', cone.sinv(lmbda, ds))
        #print('dz-tmp', dz-tmp)
        Dx, Dz = self.solve_kkt(W_mat, dx, dz - tmp)

        # Ds = W'(lambda <> ds - W Dz )
        #Ds = tmp - dot(transpose(W), dot(W, Dz))
        Ds = tmp - cone.scale(W, cone.scale(W, Dz), trans=True)
        return Dx, Dz, Ds

    def affine_direction(self, rx, rz, lmbad, W):
        # step 2: Affine direction
        # [-rx] = [0 ] + [P, G'] [dx]
        # [-rz] = [ds] + [G, 0 ] [dz]
        # lambda o (Wdz + (W^{-T})ds) = -lambda o lambda

        # TODO: add the refinement, although I don't know what it is..
        return self.Newton(W, -rx, -rz, -self.cone.sprod(lmbad, lmbad), lmbad)

    def combined_direction(self, rx, rz, lmbda, ds_aff, dz_aff, sigma, mu, W):
        # step 4: Combined direction
        # [-rx] = [0 ] + [P, G'] [dx]
        # [-rz] = [ds] + [G, 0 ] [dz]
        # lambda o (Wdz + W^{-T}ds) = -lambda o lambda - (W^{-T}ds_affine) o (W dz_affine) + sigma mu e
        cone = self.cone
        ds = - cone.sprod(lmbda, lmbda) - cone.sprod(cone.scale(W, ds_aff, trans=True, inverse=True),
                          cone.scale(W, dz_aff, trans=False, inverse=False)) + sigma * mu * cone.identity[None,:]
        return self.Newton(W, -rx, -rz, ds, lmbda)

    def __call__(self, P, q, G, h, n_l, n_Q, dim_Q, niter=None):
        """
        :param P: (b, n, n), positive semidefinite
        :param q: (b, n),
        :param G: (b, m, n)
        :param h: (b, m)
        :param n_l: number of non-negative constraints
        :param n_Q: number of second-order cone constraints
        :param dim_Q: number of second
        :return:
        """
        assert n_l + n_Q * dim_Q == h.shape[-1] == G.shape[1]
        assert P.shape[1] == P.shape[2] == q.shape[1] == G.shape[2], f"{P.shape}, {q.shape}, {G.shape}"
        # we ignore A, b here
        # we require the second order cone to be the same..
        if n_Q == 0:
            cone = self.cone = Orthant(n_l)
        elif n_l == 0:
            cone = self.cone = SecondOrder(n_Q, dim_Q)
        else:
            cone = self.cone = CartesianCone(n_l, n_Q, dim_Q)

        self.factor_kkt(P, q, G, h)

        if niter is None:
            niter = self.MAXITERS

        x, s, z = self.initialize()
        for i in range(niter):
            # step 1:
            # maybe we need to write it into a recursive form ...
            f0, rx, rz, gap, terminate = self.evaluate(x, s, z)
            if terminate.all():
                break

            # step 2: affine
            W = cone.compute_scaling(s, z) # recompute, I think we don't need to speed it up for now
            lmbda = W['lambda']
            dx_aff, dz_aff, ds_aff = self.affine_direction(rx, rz, lmbda, W)

            def inv(inv_alpha, STEP=1.):
                # get the inverse alpha, find its inverse ... and bound it into [0, 1]
                inv_alpha = torch.relu(inv_alpha)
                mask = (inv_alpha == 0).float()
                alpha = mask + (1 - mask) * STEP / inv_alpha.clamp(1e-15, np.inf)
                return alpha

            # step 3: Step size and centering paramter
            # \alpha = sup a \in [0, 1], s+a * ds_aff >=0 and z + a * dz_aff >= 0

            inv_alpha = torch.relu(torch.max(cone.max_step2(z, dz_aff), cone.max_step2(s, ds_aff)))
            alpha = inv(inv_alpha, 1).clamp(0, 1)
            sigma = (cone.sdot(s+alpha * ds_aff, z+alpha * dz_aff)/gap).clamp(0, 1) ** self.EXPON

            # step4: combined direction
            mu = gap/cone.m
            dx, dz, ds = self.combined_direction(rx, rz, lmbda, ds_aff, dz_aff, sigma, mu, W)

            # step5: update iterates and scaling matrices
            WinvTds, Wdz = cone.scale(W, ds, inverse=True, trans=True), cone.scale(W, dz)
            inv_alpha = torch.max(cone.max_step2(lmbda, WinvTds), cone.max_step2(lmbda, Wdz))
            alpha = inv(inv_alpha, self.STEP).clamp(0, 1)

            x, z, s = x + alpha * dx, z + alpha * dz, s + alpha * ds

        return x


    def evaluate(self, x, s, z):
        cone = self.cone
        vdot = cone.sdot

        # TODO: coneqp matains the gap and calculate it with lambda' * lambda
        gap = cone.sdot(s, z)

        # calcualte several values with current variables
        # [r_x] = [0] + [P, G'] [x] + [c ]
        # [r_z] = [s] + [G, 0 ] [z] + [-h]
        # mu = (s^Tz)/m, notice m is the degree of the cone

        rx = self.q + dot(self.P, x)
        f0 = 0.5 * (vdot(x, rx) + vdot(x, self.q))
        rx = rx + dot(self.GT, z)
        resx = rx.norm(dim=-1)

        rz = s + dot(self.G, x) - self.h
        resz = cone.snrm2(rz)

        # pcost = (1/2)*x'*P*x + q'*x
        # dcost = (1/2)*x'*P*x + q'*x + y'*(A*x-b) + z'*(G*x-h)
        #       = (1/2)*x'*P*x + q'*x + y'*(A*x-b) + z'*(G*x-h+s) - z'*s
        #       = (1/2)*x'*P*x + q'*x + y'*ry + z'*rz - gap
        pcost = f0
        dcost = f0 + cone.sdot(z, rz) - gap

        relgap = gap / torch.max(pcost.abs(), dcost.abs())
        # NOTE: we use some slightly different method to calculate relgap
        pres = resz/self.resz0
        dres = resx/self.resx0
        terminate = (pres <= self.feastol) & (dres <= self.feastol) & (relgap<=self.reltol)
        return f0, rx, rz, gap, terminate



class ConeQP:
    def __call__(self, P, q, G, h, n_l, n_Q, dim_Q, niter=None):
        from cvxopt import matrix
        #from cvxopt.coneprog import coneqp
        from robot.torch_robotics.solver.coneqp_python import coneqp

        device, dtype = P.device, P.dtype
        output = []
        for p, q, g, h in zip(P, q, G, h):
            # for numpy we don't need to transpose ...
            P = matrix(p.detach().cpu().numpy())
            q = matrix(q.detach().cpu().numpy())
            G = matrix(g.detach().cpu().numpy())
            h = matrix(h.detach().cpu().numpy())
            dims = {'l': n_l, 'q': [dim_Q for i in range(n_Q)], 's': []}
            x = coneqp(P, q, G, h, dims)['x']
            output.append(np.array(x)[:, 0])
        return torch.tensor(np.array(output), device=device, dtype=dtype)
