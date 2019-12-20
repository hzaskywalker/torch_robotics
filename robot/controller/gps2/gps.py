"""
BADMM GPS
"""
import numpy as np
from ...model.gmm_lr.model import GMMPriorLRDynamics
from .LQG import LQGSolver, LinearGaussianPolicy

class LQGTrajectory:
    def __init__(self, env, dynamics:GMMPriorLRDynamics):
        self.env = env
        self.dynamic = dynamics
        self.solver = LQGSolver()
        self.policy: LinearGaussianPolicy = None

    def _rollout(self):
        # Generate samples{τji}from each linear-Gaussian controllerpi(τ) by performing rollouts
        x = self.env.reset()
        it = 0

        X = []
        U = []
        while True:
            noise = np.random.normal(self.policy.dU)
            u = self.policy.act(x, it, noise)
            t, reward, done, _ = self.env.step(u)
            it += 1

            U.append(u); X.append(x)
            if done:
                break
        return np.array(X), np.array(U)

    def rollout(self, n):
        X, U = [], []
        for i in range(n):
            x, u = self._rollout()
            X.append(x)
            U.append(u)
        self.X, self.U = np.stack(X), np.stack(U)
        return self.X, self.U

    def get_data(self):
        # Prepare for
        # Minimize ∑i, tλi, tDKL(pi(xt) πθ(ut | xt)‖pi(xt, ut))with respect toθusing samples{τji}
        return self.X, self.U

    def fit(self, X, U):
        # Fit the dynamics pi(xt+1|xt,ut)to the samples{τji}
        self.dynamic.update_prior(X, U)
        self.dynamic.fit(X, U)

    def update_policy(self, eta):
        #Update pi(ut | xt) using the algorithm in Section 3 and the supplementary appendix
        # similar to
        raise NotImplementedError

    def update_lambda(self):
        raise NotImplementedError


class GPS:
    def __init__(self, model, agent, locals, world):
        self.model = model
        self.agent = agent # agent is a neural network policy
        self.locals = locals
        self.world = world

        self.buffer = []


    def update_model(self):
        raise NotImplementedError

    def iteration(self, inner_iterations):
        self.model.update(self.buffer)

        #for _ in range(inner_iterations):
        #if len(self.buffer) > 0:
        #    self.agent.udpate(self.buffer)
        #for world, traj in zip(self.world, ):
