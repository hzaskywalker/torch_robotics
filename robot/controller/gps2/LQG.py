"""
https://github.com/anassinator/ilqr for comparision
"""
import numpy as np
from typing import List
import scipy as sp
from scipy.linalg import LinAlgError
import logging
from .distributions import LinearGaussian

LOGGER = logging.getLogger(__name__)


def LQGforward(trajs: List[LinearGaussian], dynamics: List[LinearGaussian], initial: LinearGaussian):
    T, dU, dX = len(trajs), trajs[0].dY, trajs[0].dX
    # Allocate space.
    sigma = np.zeros((T, dX + dU, dX + dU))
    mu = np.zeros((T, dX + dU))

    _sigma = initial.sigma
    _mu = initial.b

    for t in range(T):
        K = trajs[t].W
        u_mu, u_sigma = trajs[t].multi(_mu, _sigma)

        sigma[t, :, :] = np.vstack([
            np.hstack([_sigma, _sigma.dot(K.T)]),
            np.hstack([K.dot(_sigma), u_sigma])
        ])
        mu[t, :] = np.hstack([_mu, u_mu])

        if t < T - 1:
            _mu, _sigma = dynamics[t].multi(mu[t], sigma[t])
    return mu, sigma

def LQGeval(trajs: List[LinearGaussian], dynamics: List[LinearGaussian], initial: LinearGaussian, l_xuxu, l_xu):
    """
    We don't evaluate the entropy term.
    As it deosn't affect the performance of the deterministic policy.
    """
    mu, sigma = LQGforward(trajs, dynamics, initial)
    #print('initial', initial.b)
    #print(mu, sigma)
    cost = 0
    for t in range(len(trajs)):
        _mu, _sigma = mu[t], sigma[t]
        cost += 0.5 * _mu.T.dot(l_xuxu[t]).dot(_mu) + l_xu[t].dot(_mu)
    return cost


def LQGbackward(dynamics: [LinearGaussian], l_xuxu, l_xu):
    """
    :param l_xuxu: (T, dU + dX, dU + dX)
    :param l_xu: (T, dU + dX)
    :return:
    """
    T = len(dynamics)
    dX = dynamics[0].b.shape[0]
    dU = dynamics[0].W.shape[1] - dX
    idx_x = slice(dX)
    idx_u = slice(dX, dX + dU)

    # Compute state-action-state function at each time step.
    Vxx, Vx = None, None
    outs = []
    for t in range(T - 1, -1, -1):
        # Add in the cost.
        f_xu = dynamics[t].W  # (X, X+U)
        f_b = dynamics[t].b  # (X,)

        Qtt = l_xuxu[t].copy()  # (X+U) x (X+U)
        Qt = l_xu[t].copy()  # (X+U,)
        if t < T - 1:
            Qtt += f_xu.T.dot(Vxx).dot(f_xu)
            Qt += f_xu.T.dot(Vx + Vxx.dot(f_b))
        Qtt = 0.5 * (Qtt + Qtt.T)

        Quu = Qtt[idx_u, idx_u]
        try:
            U = sp.linalg.cholesky(Quu)
            L = U.T
        except LinAlgError as e:
            LOGGER.debug('LinAlgError: %s', e)
            return None

        def inv_Quu(x):
            return sp.linalg.solve_triangular(
                U, sp.linalg.solve_triangular(L, x, lower=True)
            )

        traj_sigma = inv_Quu(np.eye(dU))
        k = -inv_Quu(Qt[idx_u])
        K = -inv_Quu(Qtt[idx_u, idx_x])

        Vxx = Qtt[idx_x, idx_x] + Qtt[idx_x, idx_u].dot(K)
        Vx = Qt[idx_x] + Qtt[idx_x, idx_u].dot(k)
        Vxx = 0.5 * (Vxx + Vxx.T)

        outs.append(LinearGaussian(K, k, traj_sigma, inv_sigam=Quu))
    return outs[::-1]


def soft_KL_LQG(dynamics, l_xuxu, l_xu, prev_trajs: List[LinearGaussian], eta, delta=1e-4):
    T = len(dynamics)
    eta0 = eta

    while True:
        _l_xuxu = []
        _l_xu = []

        for i in range(T):
            _logp_xuxu, _logp_xu = prev_trajs[i].log_derivative()
            _l_xuxu.append(l_xuxu[i]/eta + _logp_xuxu)
            _l_xu.append(l_xu[i]/eta + _logp_xu)

        policy = LQGbackward(dynamics, _l_xuxu, _l_xu)
        if policy is not None:
            return policy, eta

        old_eta = eta
        eta = eta0 + delta
        LOGGER.debug('Increasing eta: %f -> %f', old_eta, eta)
        delta *= 2  # Increase del_ exponentially on failure.

        if eta > 1e16:
            raise ValueError('Failed to find PD solution even for very \
                    large eta (check that dynamics and cost are \
                    reasonably well conditioned)!')


def kl_divergence(p1: List[LinearGaussian], p2: List[LinearGaussian], dynamics, initial):
    # calculate the KL divergence between two policy along the trajectory
    mu, sigma = LQGforward(p1, dynamics, initial) # trajectory distribution
    kl = 0
    for t in range(len(p1)):
        kl += p1[t].Elogp(mu[t], sigma[t]) - p2[t].Elogp(mu[t], sigma[t])
    return kl


def KL_LQG(dynamics, l_xuxu, l_xu, prev_trajs: List[LinearGaussian], epsilon: float,
           eta, min_eta, max_eta,
           max_iter, eta_delta):
    initial = dynamics[0]
    dynamics = dynamics[1:]

    def _conv_check(con, epsilon):
        """Function that checks whether dual gradient descent has converged."""
        return abs(con) < 0.1 * epsilon

    while True:
        new_policy, eta = soft_KL_LQG(dynamics, l_xuxu, l_xu, prev_trajs, eta, eta_delta)
        kl_div = kl_divergence(new_policy, prev_trajs, dynamics, initial)

        con = kl_div - epsilon
        if _conv_check(con, epsilon):
            break

        if con < 0:  # Eta was too big.
            max_eta = eta
            geom = np.sqrt(min_eta * max_eta)  # Geometric mean.
            new_eta = max(geom, 0.1 * max_eta)
            LOGGER.debug("KL: %f / %f, eta too big, new eta: %f",
                         kl_div, epsilon, new_eta)
        else:  # Eta was too small.
            min_eta = eta
            geom = np.sqrt(min_eta * max_eta)  # Geometric mean.
            new_eta = min(geom, 10.0 * min_eta)
            LOGGER.debug("KL: %f / %f, eta too small, new eta: %f",
                         kl_div, epsilon, new_eta)
        eta = new_eta
    return new_policy, eta
