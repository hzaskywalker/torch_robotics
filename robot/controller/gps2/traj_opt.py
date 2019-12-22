# trajectory optimization given the cost gradient and the simul
import copy
import numpy as np
import tqdm
from .control import adjust_epsilon_rule, kl_divergence, KL_LQG, LQGeval, LinearGaussian, rollout, initial_policy
from robot.model.gmm_linear_model import GaussianLinearModel

class ILQGTrajOpt:
    """
    We fit an initial
    """
    def __init__(self, env, initial, target, T, num_samples, model: GaussianLinearModel, epsilon=1.,
                min_epsilon_ratio=0.5, max_epsilon_ratio=3.0):
        self.env = env
        self.model = model
        self.initial = initial
        self.target = target
        self.T = T
        self.num_samples = num_samples
        self._epsilon = epsilon
        self._min_epsilon_ratio = min_epsilon_ratio
        self._max_epsilon_ratio = max_epsilon_ratio

        self.cost_old_old = self.cost_new_old = None
        self.dX, self.dU = env.dof * 2, env.dof

    def sample(self, policy):
        # rollout the current policy
        trajs = []
        costs = []
        for i in range(self.num_samples):
            xu, cost = self.rollout(policy)
            costs.append(cost)
            trajs.append(xu)
        return np.array(trajs), np.array(costs)

    def update_model(self, trajs):
        X =  trajs[:, :-1].reshape(-1, self.dX + self.dU)
        Y =  trajs[:, 1:, :self.dX].reshape(-1, self.dX)
        self.model.update(X, Y)

    def add_cost(self, samples, cost):
        ans = []
        for xu in samples:
            x_t, u_t = xu[:self.dX], xu[self.dX:]
            l_c, l_xu, l_xuxu = cost(x_t, u_t)
            l_c += 0.5 * xu.T.dot(l_xuxu).dot(xu) - l_xu.dot(xu)
            l_xu -= xu.T.dot(l_xuxu)
            ans.append([l_c, l_xu, l_xuxu])
        ans = [np.mean([j[i] for j in ans], axis=0) for i in range(3)]
        self.costs.append(ans)

    def iter(self, policy, epsilon):
        trajs, costs = self.sample(policy)
        self.update_model(trajs)

        dynamics = [copy.deepcopy(self.initial)]
        self.costs = []

        for t in range(self.T - 1):
            dynamics.append(self.model.eval(trajs[:, t], trajs[:, t+1, :self.dX]))
            self.add_cost(trajs[:, t], self.env.cost)
        self.add_cost(trajs[:, -1], self.env.cost)

        l_const, l_xu, l_xuxu = [[j[i] for j in self.costs] for i in range(3)]

        cost_new_new = LQGeval(policy, dynamics, l_xuxu, l_xu, False, l_const)

        if self.cost_old_old is not None:
            # adjust epsilon
            epsilon = adjust_epsilon_rule(epsilon, self.cost_old_old, self.cost_new_old, cost_new_new)
            epsilon = max(min(epsilon, self._max_epsilon_ratio * self._epsilon), self._min_epsilon_ratio * self._epsilon)

        old = policy
        policy, eta = KL_LQG(dynamics, l_xuxu, l_xu, policy, epsilon=epsilon)
        print('kl', kl_divergence(policy, old, dynamics))

        # calculate and save cost to adjust epsilon
        self.cost_new_old = LQGeval(policy, dynamics, l_xuxu, l_xu, False, l_const)
        self.cost_old_old = cost_new_new

        return policy, epsilon

    def rollout(self, policy):
        return rollout(policy, self.env, self.initial.sample(), self.target, self.T)

    def run(self, num_iters):
        # we fit an initial gaussian model from the data
        policy = initial_policy(self.dX, self.dU, self.T)

        epsilon = self._epsilon
        for i in tqdm.trange(num_iters):
            policy, epsilon = self.iter(policy, epsilon)

        print('END:', 'COST:', self.rollout(policy)[1])
