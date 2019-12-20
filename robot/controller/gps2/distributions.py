import scipy as sp
import numpy as np

class LinearGaussian:
    """
    Linear Gaussian Model N(Wx + b, sigma)
    W: (Y, X)
    b (Y,)
    sigma (Y, Y)

    We use this to represent both the trajectory and the
    """
    def __init__(self, W, b, sigma=None, inv_sigam=None, chol_sigma=None):
        self.W = W
        self.b = b
        self.dY, self.dX = W.shape

        if sigma is None:
            # deterministic
            sigma = np.zeros((self.dY, self.dY), dtype=np.float32)

        self.sigma = sigma
        self._inv_sigma = inv_sigam
        self._chol_sigma = chol_sigma

    @property
    def chol_sigma(self):
        if self._chol_sigma is None:
            self._chol_sigma = sp.linalg.cholesky(self.sigma)
        return self._chol_sigma


    @property
    def inv_sigma(self):
        if self._inv_sigma is None:
            U = self.chol_sigma
            L = U.T

            self._inv_sigma = sp.linalg.solve_triangular(
                U, sp.linalg.solve_triangular(L, np.eye(self.dY), lower=True)
            )
        return self._inv_sigma

    def sample(self, x):
        # sample condition on x
        return self.W.dot(x) + self.b + self.chol_sigma.T.dot(np.random.normal(size=self.dY))

    def log_derivative(self):
        """
        Let L = \log p(x, u)
        return (L_x, L_u), and ((L_xx, L_xu), (L_ux, L_uu))
        """
        K, k = self.W, self.b
        ipc = self.inv_sigma
        L_xu_xu = np.vstack([
                np.hstack([
                    K.T.dot(ipc).dot(K),
                    -K.T.dot(ipc)
                ]),
                np.hstack([
                    -ipc.dot(K), ipc
                ])
        ])
        L_xu = np.hstack([
            K.T.dot(ipc).dot(k),
            -ipc.dot(k)
        ])
        return L_xu_xu, L_xu


    def Elogp(self, mu, sigma):
        """
        NOTE: we always assuem that mu is the combination of u and x
        Thus we can estimate E_{xu~N(mu, sigma)}p(u|x)
        """
        logdet = 2 * sum(np.log(np.diag(self.chol_sigma)))
        idx_x = slice(self.dX)
        idx_u = slice(self.dX, self.dX + self.dY)
        inv_sigma = self.inv_sigma

        bias = self.b - mu[idx_u] + self.W.dot(mu[idx_x])
        return -0.5 * (
                + logdet + self.dY * np.log(2*np.pi)
                + np.trace(sigma[idx_u, idx_u].dot(inv_sigma))
                + np.trace(sigma[idx_x, idx_x].dot(self.W.T).dot(inv_sigma).dot(self.W))
                - 2 * np.trace(sigma[idx_u, idx_x].dot(self.W.T).dot(inv_sigma))
                + float(bias.T.dot(inv_sigma).dot(bias))
            )


    def logp(self, x, u):
        logdet = 2 * sum(np.log(np.diag(self.chol_sigma)))
        mu = self.W.dot(x) + self.b

        return -0.5 *(
            logdet + self.dY * np.log(2 * np.pi)
            + (u-mu).T.dot(self.inv_sigma).dot(u-mu)
        )

    def multi(self, mu, sigma):
        return self.W.dot(mu) + self.b, self.W.dot(sigma).dot(self.W.T) + self.sigma


    def __str__(self):
        return f'W: {self.W}\nb: {self.b}\nsigma: {self.sigma}'
