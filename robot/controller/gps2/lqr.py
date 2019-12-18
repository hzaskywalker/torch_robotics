import numpy as np
from .utils import check_shape

class LinearGaussianPolicy:
    """
    Time-varying linear Gaussian policy.
    U = K*x + k + noise, where noise ~ N(0, chol_pol_covar)

    pol_vovar = chol_pol_covar.dot(chol_pol_covar.T)
    """
    def __init__(self, K, k, pol_covar, chol_pol_covar, inv_pol_covar):
        Policy.__init__(self)

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
    def forward(self, traj_distr, traj_info):
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
        Fm = traj_info.dynamics.Fm
        fv = traj_info.dynamics.fv
        dyn_covar = traj_info.dynamics.dyn_covar

        # Set initial covariance (initial mu is always zero).
        sigma[0, idx_x, idx_x] = traj_info.x0sigma
        mu[0, idx_x] = traj_info.x0mu

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
                sigma[t+1, idx_x, idx_x] = \
                    Fm[t, :, :].dot(sigma[t, :, :]).dot(Fm[t, :, :].T) + \
                    dyn_covar[t, :, :]
                mu[t+1, idx_x] = Fm[t, :, :].dot(mu[t, :]) + fv[t, :]
        return mu, sigma

    def backward(self, prev_traj_distr, traj_info, eta, algorithm, m):
        """
        Perform LQR backward pass. This computes a new linear Gaussian
        policy object.
        Args:
            prev_traj_distr: A linear Gaussian policy object from
                previous iteration.
            traj_info: A TrajectoryInfo object.
            eta: Dual variable.
            algorithm: Algorithm object needed to compute costs.
            m: Condition number.
        Returns:
            traj_distr: A new linear Gaussian policy.
            new_eta: The updated dual variable. Updates happen if the
                Q-function is not PD.
        """
        # Constants.
        T = prev_traj_distr.T
        dU = prev_traj_distr.dU
        dX = prev_traj_distr.dX

        if self._update_in_bwd_pass:
            traj_distr = prev_traj_distr.nans_like()
        else:
            traj_distr = prev_traj_distr.copy()

        # Store pol_wt if necessary
        if type(algorithm) == AlgorithmBADMM:
            pol_wt = algorithm.cur[m].pol_info.pol_wt

        idx_x = slice(dX)
        idx_u = slice(dX, dX + dU)

        # Pull out dynamics.
        Fm = traj_info.dynamics.Fm
        fv = traj_info.dynamics.fv

        # Non-SPD correction terms.
        del_ = self._hyperparams['del0']
        if self.cons_per_step:
            del_ = np.ones(T) * del_
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

            if not self._update_in_bwd_pass:
                new_K, new_k = np.zeros((T, dU, dX)), np.zeros((T, dU))
                new_pS = np.zeros((T, dU, dU))
                new_ipS, new_cpS = np.zeros((T, dU, dU)), np.zeros((T, dU, dU))

            fCm, fcv = algorithm.compute_costs(
                m, eta, augment=(not self.cons_per_step)
            )

            # Compute state-action-state function at each time step.
            for t in range(T - 1, -1, -1):
                # Add in the cost.
                Qtt[t] = fCm[t, :, :]  # (X+U) x (X+U)
                Qt[t] = fcv[t, :]  # (X+U) x 1

                # Add in the value function from the next time step.
                if t < T - 1:
                    if type(algorithm) == AlgorithmBADMM:
                        multiplier = (pol_wt[t + 1] + eta) / (pol_wt[t] + eta)
                    else:
                        multiplier = 1.0
                    Qtt[t] += multiplier * \
                              Fm[t, :, :].T.dot(Vxx[t + 1, :, :]).dot(Fm[t, :, :])
                    Qt[t] += multiplier * \
                             Fm[t, :, :].T.dot(Vx[t + 1, :] +
                                               Vxx[t + 1, :, :].dot(fv[t, :]))

                # Symmetrize quadratic component.
                Qtt[t] = 0.5 * (Qtt[t] + Qtt[t].T)

                if not self.cons_per_step:
                    inv_term = Qtt[t, idx_u, idx_u]
                    k_term = Qt[t, idx_u]
                    K_term = Qtt[t, idx_u, idx_x]
                else:
                    inv_term = (1.0 / eta[t]) * Qtt[t, idx_u, idx_u] + \
                               prev_traj_distr.inv_pol_covar[t]
                    k_term = (1.0 / eta[t]) * Qt[t, idx_u] - \
                             prev_traj_distr.inv_pol_covar[t].dot(prev_traj_distr.k[t])
                    K_term = (1.0 / eta[t]) * Qtt[t, idx_u, idx_x] - \
                             prev_traj_distr.inv_pol_covar[t].dot(prev_traj_distr.K[t])
                # Compute Cholesky decomposition of Q function action
                # component.
                try:
                    U = sp.linalg.cholesky(inv_term)
                    L = U.T
                except LinAlgError as e:
                    # Error thrown when Qtt[idx_u, idx_u] is not
                    # symmetric positive definite.
                    LOGGER.debug('LinAlgError: %s', e)
                    fail = t if self.cons_per_step else True
                    break

                if self._hyperparams['update_in_bwd_pass']:
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
                else:
                    # Store conditional covariance, inverse, and Cholesky.
                    new_ipS[t, :, :] = inv_term
                    new_pS[t, :, :] = sp.linalg.solve_triangular(
                        U, sp.linalg.solve_triangular(L, np.eye(dU), lower=True)
                    )
                    new_cpS[t, :, :] = sp.linalg.cholesky(
                        new_pS[t, :, :]
                    )

                    # Compute mean terms.
                    new_k[t, :] = -sp.linalg.solve_triangular(
                        U, sp.linalg.solve_triangular(L, k_term, lower=True)
                    )
                    new_K[t, :, :] = -sp.linalg.solve_triangular(
                        U, sp.linalg.solve_triangular(L, K_term, lower=True)
                    )

                # Compute value function.
                if (self.cons_per_step or
                        not self._hyperparams['update_in_bwd_pass']):
                    Vxx[t, :, :] = Qtt[t, idx_x, idx_x] + \
                                   traj_distr.K[t].T.dot(Qtt[t, idx_u, idx_u]).dot(traj_distr.K[t]) + \
                                   (2 * Qtt[t, idx_x, idx_u]).dot(traj_distr.K[t])
                    Vx[t, :] = Qt[t, idx_x].T + \
                               Qt[t, idx_u].T.dot(traj_distr.K[t]) + \
                               traj_distr.k[t].T.dot(Qtt[t, idx_u, idx_u]).dot(traj_distr.K[t]) + \
                               Qtt[t, idx_x, idx_u].dot(traj_distr.k[t])
                else:
                    Vxx[t, :, :] = Qtt[t, idx_x, idx_x] + \
                                   Qtt[t, idx_x, idx_u].dot(traj_distr.K[t, :, :])
                    Vx[t, :] = Qt[t, idx_x] + \
                               Qtt[t, idx_x, idx_u].dot(traj_distr.k[t, :])
                Vxx[t, :, :] = 0.5 * (Vxx[t, :, :] + Vxx[t, :, :].T)

            if not self._hyperparams['update_in_bwd_pass']:
                traj_distr.K, traj_distr.k = new_K, new_k
                traj_distr.pol_covar = new_pS
                traj_distr.inv_pol_covar = new_ipS
                traj_distr.chol_pol_covar = new_cpS

            # Increment eta on non-SPD Q-function.
            if fail:
                if not self.cons_per_step:
                    old_eta = eta
                    eta = eta0 + del_
                    LOGGER.debug('Increasing eta: %f -> %f', old_eta, eta)
                    del_ *= 2  # Increase del_ exponentially on failure.
                else:
                    old_eta = eta[fail]
                    eta[fail] = eta0[fail] + del_[fail]
                    LOGGER.debug('Increasing eta %d: %f -> %f',
                                 fail, old_eta, eta[fail])
                    del_[fail] *= 2  # Increase del_ exponentially on failure.
                if self.cons_per_step:
                    fail_check = (eta[fail] >= 1e16)
                else:
                    fail_check = (eta >= 1e16)
                if fail_check:
                    if np.any(np.isnan(Fm)) or np.any(np.isnan(fv)):
                        raise ValueError('NaNs encountered in dynamics!')
                    raise ValueError('Failed to find PD solution even for very \
                            large eta (check that dynamics and cost are \
                            reasonably well conditioned)!')
        return traj_distr, eta
