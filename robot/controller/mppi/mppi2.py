"""
This implements a shooting trajectory optimization algorithm.
The closest known algorithm is perhaps MPPI and hence we stick to that terminology.
Uses a filtered action sequence to generate smooth motions.
"""

import numpy as np
from gym.utils import seeding
from ...utils.data_parallel import DataParallel


class MPPIoptimizer:
    def __init__(self, eval_function, niter=1, num_mutation=100, kappa=1., sigma=1.,
                 beta_0=1.0, beta_1=0.0, beta_2=0.0, gamma=1., inf=int(1e9), seed=1):
        self.eval_function =  eval_function
        self.niter = niter
        self.kappa = kappa
        self.num_mutation = num_mutation

        self.sigma = sigma
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.inf = inf
        self.gamma = gamma
        print("KAPPA", kappa)
        print("SIGMA", sigma)
        print("beta", beta_0, beta_1, beta_2)
        print("TASK", num_mutation)

        self.seed = seed
        self.np_random, seed = seeding.np_random(seed)

    def generate_perturbed_actions(self, base_act):
        """
        Generate perturbed actions around a base action sequence
        """
        self.np_random.seed(self.seed)
        self.seed += 1 #TODO:is seed really important here???

        sigma, beta_0, beta_1, beta_2 = self.sigma, self.beta_0, self.beta_1, self.beta_2
        #eps = np.random.normal(loc=0, scale=1.0, size=(self.num_mutation,) + base_act.shape) * sigma
        eps = np.array([
            self.np_random.normal(loc=0, scale=1.0, size=base_act.shape)
            for _ in range(self.num_mutation)])
        for i in range(2, eps.shape[1]):
            eps[:, i] = beta_0 * eps[:, i] + beta_1 * eps[:, i - 1] + beta_2 * eps[:, i - 2]
        out = base_act[None, :] + eps
        #print(eps[-1][-1])
        #exit(0)
        return out

    def score(self, rewards):
        rewards = np.asarray(rewards)
        if len(rewards.shape) == 1:
            return rewards

        assert rewards.shape[0] == self.num_mutation
        gamma = self.gamma ** np.arange(rewards.shape[1])
        R = (rewards * gamma[None, :]).sum(axis=1)
        return R

    def __call__(self, scene, base_act):
        for idx in range(self.niter):
            act = self.generate_perturbed_actions(base_act) # (num_mutation, T, action_dim)

            R = self.eval_function(scene, act) # (num_mutation, T)
            R[R != R] = self.inf
            R = self.score(R)
            S = np.exp(self.kappa * (R - np.max(R))) #(num_mutation,)

            base_act = np.sum(S[:, None, None] * act, axis=0) / (np.sum(S) + 1e-6)
        return base_act


class Rollout:
    def __init__(self, make, env_name):
        self.env = make(env_name).unwrapped #discard the other things...

    def __call__(self, s, a):
        d = len(self.env.init_qpos)
        rewards = []
        for s, a in zip(s, a):
            self.env.sim.reset()
            self.env.set_state(s[:d], s[d:])
            self.env.sim.forward()
            reward = []
            for action in a:
                _, r, _, _ = self.env.step(action)
                reward.append(r)
            rewards.append(reward)
        return np.array(rewards)


class MPPIController:
    def __init__(self, horizon, action_space,
                 env_maker, env_name, replan_period=1,
                 niter=1, num_mutation=100, kappa=1., sigma=1.,
                 num_process=1,
                 *args, **kwargs):
        # need to predefine time length
        self.horizon = horizon
        self.action_space = action_space
        self.replan_period = replan_period

        self.optimizer = MPPIoptimizer(self.eval_function, niter, num_mutation, kappa, sigma, *args, **kwargs)
        if num_process > 1:
            self.rollout = DataParallel(num_process, Rollout, env_maker, env_name)
        else:
            self.rollout = Rollout(env_maker, env_name)
        self.reset()

    def eval_function(self, x, actions):
        x = np.tile(np.array(x)[None, :], (len(actions), 1))
        return self.rollout(x, np.array(actions))

    def init_actions(self, horizon):
        return np.array([(self.action_space.high + self.action_space.low) * 0.5 for _ in range(horizon)], dtype=np.float64)

    def set_model(self, model):
        self.model = model

    def reset(self):
        # random sample may be not good
        self.prev_actions = self.init_actions(self.horizon)
        self.ac_buf = None

    def __call__(self, obs):
        # it's the state instead of obs
        if self.ac_buf is not None:
            if self.ac_buf.shape[0] > 0:
                act, self.ac_buf = self.ac_buf[0], self.ac_buf[1:]
                return act.copy()

        self.prev_actions = self.optimizer(obs, self.prev_actions)
        self.ac_buf, self.prev_actions = np.split(self.prev_actions, [self.replan_period, ])
        self.prev_actions = np.concatenate((self.prev_actions, self.init_actions(self.replan_period)))
        return self.__call__(obs)
