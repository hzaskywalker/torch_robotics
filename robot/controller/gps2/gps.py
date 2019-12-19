"""
BADMM GPS
"""
from ...model.gmm_lr.model import GMMPriorLRDynamics
from .LQG import LQGSolver, LinearGaussianPolicy

class LQGTrajectory:
    def __init__(self, env, dynamics:GMMPriorLRDynamics):
        self.env = env
        self.dynamic = dynamics
        self.solver = LQGSolver()
        self.policy: LinearGaussianPolicy = None

    def rollout(self):
        # Generate samples{τji}from each linear-Gaussian controllerpi(τ) by performing rollouts
        raise NotImplementedError

    def fit(self):
        # Fit the dynamicspi(xt+1|xt,ut)to the samples{τji}
        raise NotImplementedError

    def get_data(self):
        # Prepare for
        # Minimize ∑i, tλi, tDKL(pi(xt) πθ(ut | xt)‖pi(xt, ut))with respect toθusing samples{τji}
        raise NotImplementedError

    def update_policy(self, eta):
        #Update pi(ut | xt) using the algorithm in Section 3 and the supplementary appendix
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
