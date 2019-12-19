import numpy as np
from .GMM_prior import DynamicsPriorGMM
from .linear_dynamic import DynamicsLRPrior

class GMMPriorLRDynamics:
    def __init__(self, regularization=1e-6, max_clusters=20, min_samples_per_cluster=40, max_samples=20, initial_state_var=1e-6):
        dynamics = {
            'type': DynamicsLRPrior,
            'regularization': regularization,
            'prior': {
                'type': DynamicsPriorGMM,
                'max_clusters': max_clusters,
                'min_samples_per_cluster': min_samples_per_cluster,
                'max_samples': max_samples,
            },
        }
        self.dynamcis = DynamicsLRPrior(dynamics)
        self.initial_state_var = initial_state_var

    def update_prior(self, X, U):
        self.dynamcis.prior.update(X, U)

    def fit(self, X, U):
        """
        :param X: (n, T, dX)
        :param U: (n, T, dU)
        :return:
            Fm (T, dX, dX+DU),
            fv (T, dX),
            dyn_covar (T, dX, dX),
            x0 (dX,),
            x0sigma (dX, dX)
        """
        x0 = X[:, 0, :]
        x0mu = np.mean(x0, axis=0)
        self.x0mu = x0mu
        self.x0sigma = np.diag(
            np.maximum(np.var(x0, axis=0),
                       self.initial_state_var)
        )

        prior = self.dynamcis.get_prior()
        if prior:
            mu0, Phi, priorm, n0 = prior.initial_state()
            N = len(X)
            self.x0sigma += \
                Phi + (N * priorm) / (N + priorm) * \
                np.outer(x0mu - mu0, x0mu - mu0) / (N + n0)
        self.Fm, self.fv, self.dyn_covar = self.dynamcis.fit(X, U)
        return self.Fm, self.fv, self.dyn_covar, self.x0mu, self.x0sigma
