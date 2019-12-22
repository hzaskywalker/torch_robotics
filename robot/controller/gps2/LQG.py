import numpy as np
from typing import List
import scipy as sp
from scipy.linalg import LinAlgError
import logging
from robot.utils.distributions import LinearGaussian

LOGGER = logging.getLogger(__name__)


def LQGforward(trajs: List[LinearGaussian], dynamics: List[LinearGaussian]):
    T, dU, dX = len(trajs), trajs[0].dY, trajs[0].dX
    # Allocate space.
    sigma = np.zeros((T, dX + dU, dX + dU))
    mu = np.zeros((T, dX + dU))

    _sigma = dynamics[0].sigma
    _mu = dynamics[0].b

    for t in range(T):
        K = trajs[t].W
        u_mu, u_sigma = trajs[t].multi(_mu, _sigma)

        sigma[t, :, :] = np.vstack([
            np.hstack([_sigma, _sigma.dot(K.T)]),
            np.hstack([K.dot(_sigma), u_sigma])
        ])
        mu[t, :] = np.hstack([_mu, u_mu])

        if t < T - 1:
            _mu, _sigma = dynamics[t+1].multi(mu[t], sigma[t])
    return mu, sigma


def policy_entropy(policy: List[LinearGaussian], dynamics: List[LinearGaussian]):
    idx_x = slice(policy[0].dX)
    mu, sigma = LQGforward(policy, dynamics)

    #initial.W = initial.W * 0
    #ent = initial.Elogp(mu[0, idx_x], sigma[0, idx_x, idx_x])
    # NOTE: we ignore the entropy of the first item
    ent = 0

    for t in range(len(policy)):
        ent += policy[t].Elogp(mu[t], sigma[t])
    for t in range(len(policy) - 1):
        _mu, _sigma = mu[t], sigma[t]

        K = dynamics[t].W
        u_mu, u_sigma = dynamics[t+1].multi(_mu, _sigma)

        _sigma = np.vstack([
            np.hstack([_sigma, _sigma.dot(K.T)]),
            np.hstack([K.dot(_sigma), u_sigma])
        ])
        _mu = np.hstack([_mu, u_mu])
        ent += dynamics[t+1].Elogp(_mu, _sigma)
    return ent


def LQGeval(trajs: List[LinearGaussian], dynamics: List[LinearGaussian], l_xuxu, l_xu, entropy=False, l_const=None, deterministic=0):
    mu, sigma = LQGforward(trajs, dynamics)
    cost = 0

    for t in range(len(trajs)):
        _mu = mu[t]
        cost += 0.5 * _mu.T.dot(l_xuxu[t]).dot(_mu) + l_xu[t].dot(_mu) \
                + 0.5 * np.trace(sigma[t].dot(l_xuxu[t])) * (1-deterministic)
        if l_const is not None:
            cost += l_const[t]
    if entropy is True:
        cost += policy_entropy(trajs, dynamics)
    return cost


def LQGbackward(dynamics: [LinearGaussian], l_xuxu, l_xu):
    """
    :param dyanmics: linearize it at f_xu * xu + f_b
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
        Qtt = l_xuxu[t].copy()  # (X+U) x (X+U)
        Qt = l_xu[t].copy()  # (X+U,)

        if t < T - 1:
            f_xu = dynamics[t + 1].W  # (X, X+U)
            f_b = dynamics[t + 1].b  # (X,)
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


def soft_KL_LQG(dynamics, l_xuxu, l_xu, prev_traj: List[LinearGaussian], eta, delta=1e-4):
    T = len(dynamics)
    eta0 = eta

    while True:
        _l_xuxu = []
        _l_xu = []

        for i in range(T):
            _logp_xuxu, _logp_xu = prev_traj[i].log_derivative()
            assert _logp_xuxu.shape == l_xuxu[i].shape
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


def kl_divergence(p1: List[LinearGaussian], p2: List[LinearGaussian], dynamics):
    # calculate the KL divergence between two policy along the trajectory
    mu, sigma = LQGforward(p1, dynamics) # trajectory distribution
    kl = 0
    for t in range(len(p1)-1):
        kl += p1[t].Elogp(mu[t], sigma[t]) - p2[t].Elogp(mu[t], sigma[t])
    return kl


def KL_LQG(dynamics, l_xuxu, l_xu, prev_trajs: List[LinearGaussian], epsilon: float,
           eta=1., min_eta=1e-8, max_eta=1e16,
           max_iter=20, eta_delta=1e-4):

    def _conv_check(con, epsilon):
        """Function that checks whether dual gradient descent has converged."""
        return abs(con) < 0.1 * epsilon

    assert max_iter > 0
    new_policy = None

    for i in range(max_iter):
        new_policy, eta = soft_KL_LQG(dynamics, l_xuxu, l_xu, prev_trajs, eta, eta_delta)
        kl_div = kl_divergence(new_policy, prev_trajs, dynamics)

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

def adjust_epsilon_rule(epsilon, l_old_old, l_new_old, l_new_new):
    multi = (l_old_old - l_new_old)/(2 * max(1e-4, l_new_new - l_new_old) )
    multi = max(0.1, min(5.0, multi))
    return epsilon * multi
