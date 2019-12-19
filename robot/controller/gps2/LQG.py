"""
https://github.com/anassinator/ilqr for comparision
"""
import numpy as np
import scipy as sp
from .utils import check_shape
from scipy.linalg import LinAlgError
import logging

LOGGER = logging.getLogger(__name__)

class LinearGaussianPolicy:
    """
    Time-varying linear Gaussian policy.
    U = K*x + k + noise, where noise ~ N(0, chol_pol_covar)

    pol_vovar = chol_pol_covar.dot(chol_pol_covar.T)
    """
    def __init__(self, K, k, pol_covar, chol_pol_covar, inv_pol_covar):
        # Assume K has the correct shape, and make sure others match.
        self.T = K.shape[0]
        self.dU = K.shape[1]
        self.dX = K.shape[2]

        check_shape(k, (self.T, self.dU))
        check_shape(pol_covar, (self.T, self.dU, self.dU))
        check_shape(chol_pol_covar, (self.T, self.dU, self.dU))
        check_shape(inv_pol_covar, (self.T, self.dU, self.dU))

        self.K = K
        self.k = k
        self.pol_covar = pol_covar
        self.chol_pol_covar = chol_pol_covar
        self.inv_pol_covar = inv_pol_covar

    def act(self, x, obs, t, noise=None):
        """
        Return an action for a state.
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: Action noise. This will be scaled by the variance.
        """
        u = self.K[t].dot(x) + self.k[t]
        u += self.chol_pol_covar[t].T.dot(noise)
        return u

    def nans_like(self):
        """
        Returns:
            A new linear Gaussian policy object with the same dimensions
            but all values filled with NaNs.
        """
        policy = LinearGaussianPolicy(
            np.zeros_like(self.K), np.zeros_like(self.k),
            np.zeros_like(self.pol_covar), np.zeros_like(self.chol_pol_covar),
            np.zeros_like(self.inv_pol_covar)
        )
        policy.K.fill(np.nan)
        policy.k.fill(np.nan)
        policy.pol_covar.fill(np.nan)
        policy.chol_pol_covar.fill(np.nan)
        policy.inv_pol_covar.fill(np.nan)
        return policy


class LQGSolver:
    def __init__(self):
        self._update_in_bwd_pass =True  # Whether or not to update the TVLG controller during the bwd pass.
        self._del0 = 1e-4

        self._cons_per_step = False  # Whether or not to enforce separate KL constraints at each time step.
        self._use_prev_distr = False  # Whether or not to measure expected KL under the previous traj distr.

    def forward(self, traj_distr, dynamics, initial):
        """
        Perform LQR forward pass. Computes state-action marginals from
        dynamics and policy.
        Args:
            traj_distr: A linear Gaussian policy object.
            traj_info: A TrajectoryInfo object.
        Returns:
            mu: A T x dX mean action vector.
            sigma: A T x dX x dX covariance matrix.
        """
        # Compute state-action marginals from specified conditional
        # parameters and current traj_info.
        T = traj_distr.T
        dU = traj_distr.dU
        dX = traj_distr.dX

        # Constants.
        idx_x = slice(dX)

        # Allocate space.
        sigma = np.zeros((T, dX+dU, dX+dU))
        mu = np.zeros((T, dX+dU))

        # Pull out dynamics.
        Fm = dynamics.Fm # scalar factor
        fv = dynamics.fv #bias term
        dyn_covar = dynamics.dyn_covar

        # Set initial covariance (initial mu is always zero).
        sigma[0, idx_x, idx_x] = initial.x0sigma
        mu[0, idx_x] = initial.x0mu

        for t in range(T):
            sigma[t, :, :] = np.vstack([
                np.hstack([
                    sigma[t, idx_x, idx_x],
                    sigma[t, idx_x, idx_x].dot(traj_distr.K[t, :, :].T)
                ]),
                np.hstack([
                    traj_distr.K[t, :, :].dot(sigma[t, idx_x, idx_x]),
                    traj_distr.K[t, :, :].dot(sigma[t, idx_x, idx_x]).dot(
                        traj_distr.K[t, :, :].T
                    ) + traj_distr.pol_covar[t, :, :]
                ])
            ])
            mu[t, :] = np.hstack([
                mu[t, idx_x],
                traj_distr.K[t, :, :].dot(mu[t, idx_x]) + traj_distr.k[t, :]
            ])
            if t < T - 1:
                # calculate the forward model
                sigma[t+1, idx_x, idx_x] = \
                    Fm[t, :, :].dot(sigma[t, :, :]).dot(Fm[t, :, :].T) + \
                    dyn_covar[t, :, :]
                mu[t+1, idx_x] = Fm[t, :, :].dot(mu[t, :]) + fv[t, :]
        return mu, sigma

    def augment(self, Cm, cv, traj_distr, eta, max_ent_traj):
        multiplier = max_ent_traj
        fCm, fcv = Cm / (eta + multiplier), cv / (eta + multiplier)
        T, K, ipc, k = traj_distr.T, traj_distr.K, traj_distr.inv_pol_covar, traj_distr.k

        # Add in the trajectory divergence term.
        for t in range(T - 1, -1, -1):
            fCm[t, :, :] += eta / (eta + multiplier) * np.vstack([
                np.hstack([
                    K[t, :, :].T.dot(ipc[t, :, :]).dot(K[t, :, :]),
                    -K[t, :, :].T.dot(ipc[t, :, :])
                ]),
                np.hstack([
                    -ipc[t, :, :].dot(K[t, :, :]), ipc[t, :, :]
                ])
            ])
            fcv[t, :] += eta / (eta + multiplier) * np.hstack([
                K[t, :, :].T.dot(ipc[t, :, :]).dot(k[t, :]),
                -ipc[t, :, :].dot(k[t, :])
            ])

        return fCm, fcv

    def backward(self, traj_distr, dynamics, eta, cost_func):
        """
        Perform LQR backward pass. This computes a new linear Gaussian
        policy object.
        Args:
            prev_traj_distr: A linear Gaussian policy object from
                previous iteration.
            dynamics: Dynamics, Fm.dot(mu) + fv
            eta: Dual variable.
            cost_func: Algorithm object needed to compute costs.
        Returns:
            traj_distr: A new linear Gaussian policy.
            new_eta: The updated dual variable. Updates happen if the
                Q-function is not PD.
        """
        # Constants.
        T = traj_distr.T
        dU = traj_distr.dU
        dX = traj_distr.dX

        idx_x = slice(dX)
        idx_u = slice(dX, dX + dU)

        # Pull out dynamics.
        Fm = dynamics.Fm
        fv = dynamics.fv

        # Non-SPD correction terms.
        del_ = self._del0
        eta0 = eta

        # Run dynamic programming.
        fail = True
        while fail:
            fail = False  # Flip to true on non-symmetric PD.

            # Allocate.
            Vxx = np.zeros((T, dX, dX))
            Vx = np.zeros((T, dX))
            Qtt = np.zeros((T, dX + dU, dX + dU))
            Qt = np.zeros((T, dX + dU))

            fCm, fcv = cost_func(m, eta, augment=(not self._cons_per_step))

            # Compute state-action-state function at each time step.
            for t in range(T - 1, -1, -1):
                # Add in the cost.
                Qtt[t] = fCm[t, :, :]  # (X+U) x (X+U), l_{xu, xu}
                Qt[t] = fcv[t, :]  # (X+U) x 1, l_{xu}

                # Add in the value function from the next time step.
                if t < T - 1:
                    #if type(algorithm) == AlgorithmBADMM:
                    #    multiplier = (pol_wt[t + 1] + eta) / (pol_wt[t] + eta)
                    #else:
                    # hza: update Q, the same as the paper
                    multiplier = 1.0
                    Qtt[t] += multiplier * \
                              Fm[t, :, :].T.dot(Vxx[t + 1, :, :]).dot(Fm[t, :, :])
                    Qt[t] += multiplier * \
                             Fm[t, :, :].T.dot(Vx[t + 1, :] +
                                               Vxx[t + 1, :, :].dot(fv[t, :]))

                # Symmetrize quadratic component.
                Qtt[t] = 0.5 * (Qtt[t] + Qtt[t].T)

                inv_term = Qtt[t, idx_u, idx_u]
                k_term = Qt[t, idx_u]
                K_term = Qtt[t, idx_u, idx_x]

                try:
                    U = sp.linalg.cholesky(inv_term)
                    L = U.T
                except LinAlgError as e:
                    # Error thrown when Qtt[idx_u, idx_u] is not
                    # symmetric positive definite.
                    LOGGER.debug('LinAlgError: %s', e)
                    fail = t if self._cons_per_step else True
                    break

                # Store conditional covariance, inverse, and Cholesky.
                traj_distr.inv_pol_covar[t, :, :] = inv_term
                traj_distr.pol_covar[t, :, :] = sp.linalg.solve_triangular(
                    U, sp.linalg.solve_triangular(L, np.eye(dU), lower=True)
                )
                traj_distr.chol_pol_covar[t, :, :] = sp.linalg.cholesky(
                    traj_distr.pol_covar[t, :, :]
                )

                # Compute mean terms.
                traj_distr.k[t, :] = -sp.linalg.solve_triangular(
                    U, sp.linalg.solve_triangular(L, k_term, lower=True)
                )
                traj_distr.K[t, :, :] = -sp.linalg.solve_triangular(
                    U, sp.linalg.solve_triangular(L, K_term, lower=True)
                )

                Vxx[t, :, :] = Qtt[t, idx_x, idx_x] + \
                               Qtt[t, idx_x, idx_u].dot(traj_distr.K[t, :, :])
                Vx[t, :] = Qt[t, idx_x] + \
                           Qtt[t, idx_x, idx_u].dot(traj_distr.k[t, :])
                Vxx[t, :, :] = 0.5 * (Vxx[t, :, :] + Vxx[t, :, :].T)

            # Increment eta on non-SPD Q-function.
            if fail:
                old_eta = eta
                eta = eta0 + del_
                LOGGER.debug('Increasing eta: %f -> %f', old_eta, eta)
                del_ *= 2  # Increase del_ exponentially on failure.

                fail_check = (eta >= 1e16)

                if fail_check:
                    if np.any(np.isnan(Fm)) or np.any(np.isnan(fv)):
                        raise ValueError('NaNs encountered in dynamics!')
                    raise ValueError('Failed to find PD solution even for very \
                            large eta (check that dynamics and cost are \
                            reasonably well conditioned)!')
        return traj_distr, eta
