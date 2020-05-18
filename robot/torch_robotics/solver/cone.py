import torch

class Cone:
    _identity = None
    _m = None

    @property
    def identity(self):
        return self._identity

    @property
    def m(self):
        return self._m

    def positive(self, x):
        raise NotImplementedError

    def inside(self, x):
        raise NotImplementedError

    def max_step(self, x):
        """
        Returns min {t | x + t*e >= 0}, where e is defined as follows

        - For the nonlinear and 'l' blocks: e is the vector of ones.
        - For the 'q' blocks: e is the first unit vector.
        - For the 's' blocks: e is the identity matrix.
        """
        raise NotImplementedError

    def max_step2(self, a, b):
        # return inf {\alpha|\alpha a+b >= 0 }
        b2 = self.scale2(a, b)
        return self.max_step(b2)

    def sprod(self, x, y):
        # Return x o y
        raise NotImplementedError

    def sinv(self, x, y):
        # Return x o\ y or x <> y in the book
        raise NotImplementedError

    def sdot(self, x, y):
        # for both l and Q, the |xoy| is just the l2 norm
        return (x*y).sum(dim=-1)

    def snrm2(self, x):
        # return the norm ... I don't know why it's called snrm2
        return x.norm(dim=-1)

    def scale2(self, lmbda, x, inverse=False):
        """
        Evaluates

            x := H(lambda^{1/2}) * x   (inverse is 'N')
            x := H(lambda^{-1/2}) * x  (inverse is 'I').

            Notice that if lambda = w, then W = H(lambda^{-1/2})

        H is the Hessian of the logarithmic barrier.
        """
        raise NotImplementedError

    def compute_scaling(self, s, z):
        """
        Returns the Nesterov-Todd scaling W at points s and z, and stores the
        scaled variable in lmbda.

            W * z = W^{-T} * s = lmbda.

        """
        raise NotImplementedError

    def as_matrix(self, W, trans=False, inverse=False):
        raise NotImplementedError

    def scale(self, W, x, trans=False, inverse=False):
        """
        Nesterov-Todd scaling ...
        x := W * x          (trans is False, inverse='N')
        x := W ^ T * x      (trans is True,  inverse='N')
        x := W ^ {-1} * x   (trans is False, inverse='I')
        x := W ^ {-T} * x   (trans is True,  inverse='I').
        """
        raise NotImplementedError


class Orthant(Cone):
    def __init__(self, n, device='cuda:0'):
        self._identity = torch.ones((n,), device=device)
        self._list = torch.arange(n, device=device)
        self._m = n

    def sprod(self, a, b):
        # u o v = (u_1v_1,\dots, u_pv_p)
        return a * b

    def sinv(self, x, y):
        # For the nonlinear and 'l' blocks:
        #
        #     xk <> yk = yk o\ xk = yk .\ xk.
        return y / x

    def compute_scaling(self, s, z):
        assert s.shape[-1] == self._m
        a = torch.sqrt(s)
        b = torch.sqrt(z)
        w = a/b
        return {'w': w, 'wi': 1/w, 'lambda': self.sprod(a, b)}

    def as_matrix(self, W, trans=False, inverse=False):
        w = W['w'] if not inverse else W['wi']
        W = w.new_zeros((w.shape[0], w.shape[1], w.shape[1]))
        W[:, self._list, self._list] = w
        return W

    def scale(self, W, x, trans=False, inverse=False):
        w = W['w'] if not inverse else W['wi']
        return w * x

    def scale2(self, lmbda, x, inverse=False):
        # For the nonlinear and 'l' blocks,
        #
        #     xk := xk ./ l   (inverse is 'N')
        #     xk := xk .* l   (inverse is 'I')
        #
        # where l is lmbda.
        if not inverse:
            return x / lmbda
        else:
            return x * lmbda

    def max_step(self, x):
        return -x.min(dim=-1)[0]

    def inside(self, x):
        return (x>=0).all(dim=-1)


class SecondOrder(Cone):
    # W, W^T, W^{-1}, W^{-T}
    # W^TW
    # provide the functions
    def __init__(self, n, dim, device='cuda:0', dtype=torch.float64):
        self.n = n
        self.dim = dim
        self._m = self.n
        self._identity = torch.zeros((self.n * self.m), device=device, dtype=dtype)
        self._identity[0::self.dim] = 1

    def sprod(self, x, y):
        x = x.reshape(-1, self.dim)
        y = x.reshape(-1, self.dim)
        return torch.cat(((x*y).sum(dim=-1, keepdims=True),
                          x[..., 0:1] * y[..., 1:] + y[..., 0:1] * x[..., 0:1]), dim=-1).reshape(-1, self.n * self.dim)

    def sinv(self, x, y):
        # x <> y = y o\ x
        # x o (x<>y) = y for all x
        # For the 'q' blocks:
        #
        #                        [ l0   -l1'              ]
        #     yk o\ xk = 1/a *   [                        ] * xk
        #                        [ -l1  (a*I + l1*l1')/l0 ]
        #
        # where yk = (l0, l1) and a = l0^2 - l1'*l1.
        x = x.reshape(-1, self.dim)
        y = x.reshape(-1, self.dim)
        l0 = y[..., 0]
        l1 = y[..., 1:]
        l1_sum = l1.sum(dim=-1)
        aa = (l0 + l1_sum) * (l0 - l1_sum) # jnrm2

        # out[0] = l0 * xk[0] - l1 * xk[1:]
        out = torch.zeros_like(x)
        cc = x[..., 0]
        ee = x[..., 1:]
        dd = (l1 * x[..., 1:]).sum(dim=-1)
        out[..., 0] = cc * l0 - dd

        # out[i] = -l1[i] * xk[0] + (a * xk[i] + l1[i] * (l1' * x[1:])) / l0
        # out[i] = -l1[i] * xk[0] + xk[i] * a/l0 + l1[i] * dd/l0
        # out = a/l0  * xk + l1[i] * (dd/l0 - cc)
        out[..., 1:] = ((dd/l0)[..., None] - cc) * l1 + aa/l0 * ee
        return out.reshape(-1, self.n * self.dim)

    def jnrm2(self, x):
        l0 = x[..., 0]
        l1 = x[..., 1:].sum(dim=-1)
        return torch.sqrt(torch.relu((l0 + l1) * (l0 - l1)))

    def compute_scaling(self, s, z):
        # For the 'q' blocks, compute lists 'v', 'beta'.
        #
        # The vector v[k] has unit hyperbolic norm:
        #     (sqrt( v[k]' * J * v[k] ) = 1 with J = [1, 0; 0, -I]).
        # beta[k] is a positive scalar.
        #
        # The hyperbolic Householder matrix H = 2*v[k]*v[k]' - J
        # defined by v[k] satisfies
        #
        #     (beta[k] * H) * zk  = (beta[k] * H) \ sk = lambda_k
        #
        # where sk = s[indq[k]:indq[k+1]], zk = z[indq[k]:indq[k+1]].
        #
        # lambda_k is stored in lmbda[indq[k]:indq[k+1]].

        W = {}
        s, z = s.reshape(-1, self.dim), z.reshape(-1, self.dim)
        aa = self.jnrm2(s)
        bb = self.jnrm2(z)
        # w_k^TJw_k
        W['beta'] = torch.sqrt(aa/bb)
        # w_ is the \bar{w_k} in the book
        s_ = s/aa
        z_ = z/bb
        cc = torch.sqrt((1 + self.sdot(z_, s_))/2) # gamma in the book
        z_[..., 0] = -z_[..., 0] # -Jz
        w_ = (s_ - z_)/cc/2.

        # v is w_^{1/2} = 1/(2(w_0+1))**0.5 * (w_+e)
        w_[:, 0] += 1.
        W['v'] = w_/torch.sqrt(2.0 * w_[:, 0])

        # To get the scaled variable lambda_k
        #
        #     d =  sk0/a + zk0/b + 2*c
        #     lambda_k = [ c;
        #                  (c + zk0/b)/d * sk1/a + (c + sk0/a)/d * zk1/b ]
        #     lambda_k *= sqrt(a * b)
        # by the book, the noramlized
        #   lambda_0 = gamma
        #   lambda_1 = 1/d * ((gamma + z_0) * s_1+(gamma+z_1) * z_1)
        dd = 2*cc + s_[:, 0] + z_[:, 0]
        lmbda = torch.zeros_like(s_)
        lmbda[:, 0] = cc
        lmbda[:, 1:] = ((cc+z_[:, 0])[:, None] * s_[:, 1:] + (cc+s_[:, 0])[:, None] * z_[:, 1:])/dd

        W['lambda'] = lmbda
        return W

    def scale(self, W, x, trans=False, inverse=False):
        # the original python code support the case that x is a matrix..
        # currently we ignore it ...

        # Scaling for 'q' component is
        #
        #     xk := beta * (2*v*v' - J) * xk
        #         = beta * (2*v*(xk'*v)' - J*xk)
        #
        # where beta = W['beta'][k], v = W['v'][k], J = [1, 0; 0, -I].
        #
        # Inverse scaling is
        #
        #     xk := 1/beta * (2*J*v*v'*J - J) * xk
        #         = 1/beta * (-J) * (2*v*((-J*xk)'*v)' + xk).

        # By the book, \bar{W_k} = 2*v*v'-J, and beta is the normalizer
        x = x.reshape(-1, self.dim)
        v = W['v']
        beta = W['beta']

        Jx = x.clone()
        Jx[:, 1:] *= -1  # Jx

        if not inverse:
            out = beta * (2 * v * self.sdot(x, v)[:, None] - Jx)
        else:
            out = 2 * v * self.sdot(-Jx, v)[:, None] + x
            out[:, 0] *= -1
        return out / beta

    def as_matrix(self, W, trans=False, inverse=False):
        def out_product(x):
            # return x * x'
            return x[:, :, None] @ x[:, None, :]
        v = W['v']
        J = torch.eye(self.dim, device=v.device, dtype=v.dtype)
        J[:, 1:] *= -1
        if inverse:
            Jv = v.clone()
            Jv[:, 1] *= -1
            out = 2 * out_product(Jv) - J
        else:
            out = 2 * out_product(v) - J
        out = out/W['beta']
        out = out.reshape(-1, self.n, self.dim, self.dim)
        if self.n == 1:
            return out[:, 0]
        else:
            out2 = out.new_zeros(out.shape[0], self.n * self.dim, self.n * self.dim)
            for i in range(self.n):
                sl = slice(i*self.dim, i*self.dim+self.dim)
                out2[:, sl, sl] = out[:, i]
            return out2

    def scale2(self, lmbda, x, inverse=False):
        raise NotImplementedError


class CartesianCone:
    def __init__(self, n_l, n_Q, dim_Q):
        raise NotImplementedError