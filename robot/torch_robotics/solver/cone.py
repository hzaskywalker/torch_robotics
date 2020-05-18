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

    def sinv(self, a, b):
        # For the nonlinear and 'l' blocks:
        #
        #     yk o\ xk = yk .\ xk.
        return a / b

    def compute_scaling(self, s, z):
        a = torch.sqrt(s)
        b = torch.sqrt(z)
        w = a/b
        return {'w': w, 'wi': 1/w, 'lambda': self.sprod(a, b)}

    def as_matrix(self, W, trans=False, inverse=False):
        w = W[0] if not inverse else W[1]
        W = w.new_zeros((w.shape[0], w.shape[1], w.shape[1]))
        W[:, self._list, self._list] = w
        return W

    def scale2(self, lmbda, x, inverse=False):
        # For the nonlinear and 'l' blocks,
        #
        #     xk := xk ./ l   (inverse is 'N')
        #     xk := xk .* l   (inverse is 'I')
        #
        # where l is lmbda.
        if inverse:
            return x / lmbda
        else:
            return x * lmbda

    def max_step(self, x):
        return x.min(dim=-1)[0]

    def inside(self, x):
        return (x>=0).all(dim=-1)


class SecondOrder(Cone):
    # W, W^T, W^{-1}, W^{-T}
    # W^TW
    # provide the functions
    def __init__(self, n, dim):
        raise NotImplementedError

class CartesianCone:
    def __init__(self, n_l, n_Q, dim_Q):
        raise NotImplementedError
