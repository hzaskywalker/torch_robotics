""" This file defines linear regression with an gaussian prior. """
import numpy as np
from robot.utils.distributions import LinearGaussian
from sklearn.mixture import GaussianMixture

def gauss_fit_joint_prior(X, mu_0, Psi, m, n_0, dX, sig_reg):
    """ Perform Gaussian fit to data with a prior. """
    """ see inverse wishart part of https://en.wikipedia.org/wiki/Multivariate_normal_distribution """
    """ https://en.wikipedia.org/wiki/Normal-inverse-Wishart_distribution for details, note m is the lambda here"""
    # Build weights matrix.
    n, d = X.shape
    assert Psi.shape[0] == d

    # Compute empirical mean and covariance.
    x = X.mean(axis=0)
    S = (X-x).T.dot(X-x) / n

    sigma = (Psi + n * S + (n * m) / (n + m) *
             np.outer(x - mu_0, x - mu_0)) / (n + n_0) # Expectation of IW distribution, Actually it should b n_0-d-1
    sigma = 0.5 * (sigma + sigma.T)

    # Add sigma regularization.
    sigma += sig_reg

    # fd best estimate the variance of the remaining elements...
    fd = np.linalg.solve(sigma[:dX, :dX], sigma[:dX, dX:d]).T
    fc = x[dX:d] - fd.dot(x[:dX])

    dynsig = sigma[dX:d, dX:d] - fd.dot(sigma[:dX, :dX]).dot(fd.T)
    dynsig = 0.5 * (dynsig + dynsig.T)
    return fd, fc, dynsig


class GMMPrior:
    def __init__(self, max_samples, max_clusters, min_samples_per_cluster):
        self.max_samples = max_samples
        self.max_clusters = max_clusters
        self.min_samples_per_cluster = min_samples_per_cluster

        self.data = None
        self.gmm: GaussianMixture = None

    def update(self, X):
        if self.data is None:
            self.data = X
        else:
            self.data = np.concatenate((self.data, X), axis=0)
        start = max(0, self.data.shape[0] - self.max_samples + 1)
        self.data = self.data[start:]
        K = int(max(2, min(self.max_clusters,
                           np.floor(float(len(X)) / self.min_samples_per_cluster))))

        #self.gmm.update(self.data, K)
        self.gmm = GaussianMixture(K, max_iter=100)
        self.gmm.fit(self.data)
        self.N = len(self.data)

    def moments(self, wts, mu, sigma):
        # wts: (K,)
        mu = np.sum(mu * wts[:, None], axis=0)

        # Compute overall covariance; the original implementation is wrong!!!
        diff = mu - np.expand_dims(mu, axis=0)
        diff_expand = diff[:, None, :] * diff[:, :, None]

        sigma = np.sum((sigma + diff_expand) * wts[:, None, None], axis=0)
        return mu, sigma

    def inference(self, pts):
        wts = self.gmm.predict_proba(pts).mean(axis=0)
        mu_0, Psi = self.moments(wts, self.gmm.means_, self.gmm.covariances_)

        # Set hyperparameters.
        m = self.N
        # usually it should be -1 but not -2, but it doesn't matter as the scale of the sigma matrix will only affect
        # the kl divergence but not the optimal control.... in some sense it's really tricky..

        n_0 = m - Psi.shape[0] - 2
        Psi *= self.N #empirical std of the data
        return mu_0, Psi, m, n_0

    def eval(self, pts, strength):
        mu_0, Psi, m, n_0 =  self.inference(pts)
        return mu_0, Psi/self.N * strength, m/self.N * strength, n_0/self.N * strength


class GaussianLinearModel:
    """ Dynamics with linear regression, with arbitrary prior. """
    def __init__(self, regularization, prior_strength, *args, **kwargs):
        self.regularization = regularization
        self.prior_strength = prior_strength
        self.prior = GMMPrior(*args, **kwargs)

    def update(self, X, Y):
        self.prior.update(np.concatenate((X, Y), axis=1))

    def eval(self, X, Y):
        """
        :param X: (N, d1)
        :param Y: (N, d2)
        :return:
        """
        """ Fit dynamics, indeed an online learning step """
        N, dX = X.shape
        dY = Y.shape[1]

        if N == 1:
            raise ValueError("Cannot fit dynamics on 1 sample")

        XY = np.concatenate((X, Y), axis=1)

        # Obtain Normal-inverse-Wishart prior.
        mu_0, Psi, mm, n_0 = self.prior.eval(XY, self.prior_strength)

        sig_reg = np.zeros((dX + dY, dX + dY))
        sig_reg[:dX, :dX] = self.regularization
        Fm, fv, dyn_covar = gauss_fit_joint_prior(XY, mu_0, Psi, mm, n_0, dX, sig_reg)
        return LinearGaussian(Fm, fv, dyn_covar)
