# Control, given an environment that support reset
import copy
import numpy as np
import tqdm
from robot.controller.gps2 import LinearGaussian, KL_LQG, LQGeval
from robot.controller.gps2.LQG import LQGforward, adjust_epsilon_rule, kl_divergence
from robot.envs.arms.env import Env

def initial_policy(dX, dU, N):
    policy = []
    for i in range(N):
        policy.append(
            LinearGaussian(np.zeros((dU, dX)), np.zeros(dU), np.eye(dU) * 0.01)
            #LinearGaussian(np.zeros((dU, dX)), np.zeros(dU), np.eye(dU) * 0.0, chol_sigma=np.zeros((dU, dU)))
        )
    return policy


def rollout(policy, env: Env, start, target, T):
    x = np.array(start)
    cost = 0
    xu = []
    for i in range(T):
        u = policy[i].sample(x)
        xu.append([x, u])

        if i!= T-1:
            l, _, _ = env.cost(x, u)
            cost += l

            x = np.array(env.forward(x, u))

    final_cost = env.cost_final(x, target)[0]

    print('start', env.arm.position(start)[:, -1])
    print('reached', env.arm.position(x)[:, -1])
    print('target', target)

    cost += final_cost
    return xu, cost


def LQR_control(env, initial: LinearGaussian, target: np.ndarray, T, num_iters, epsilon=1.,
                min_epsilon_ratio=0.5, max_epsilon_ratio=3.0):
    # hope to reach the target from the initial position to the target position.
    _epsilon = epsilon
    dX, dU = env.dof * 2, env.dof
    policy = initial_policy(dX, dU, T)

    cost_old_old = cost_new_old = None

    for i in tqdm.trange(num_iters + 1):
        # rollout
        xu, cost = rollout(policy, env, initial.sample(), target, T)
        print('ITERATION:', i, 'COST:', cost, "EPSILON", epsilon)
        if i == num_iters:
            break

        dynamics = [copy.deepcopy(initial)]
        l_xuxu = []
        l_xu = []
        l_const = []
        for t in range(T-1):
            # build new dynamics
            x_t, u_t = xu[t]
            A, B = env.finite_differences(x_t, u_t)
            # x(t+1) = f(x(t), u(t)) = A(x -^x) + B(u-^u) + x_{t+1}
            dynamics.append(
                LinearGaussian(np.concatenate((A, B), axis=1), A.dot(-x_t) + B.dot(-u_t) + xu[t+1][0])
            )

            _l_c , _l_xu , _l_xuxu = env.cost(x_t, u_t)
            l_xuxu.append(_l_xuxu)
            l_xu.append(_l_xu)
            l_const.append(_l_c)

        _final_c , _final_l_x, _final_l_xx = env.cost_final(xu[-1][0], target)
        l_xuxu.append(_final_l_xx)
        l_xu.append(_final_l_x)
        l_const.append(_final_c)

        for t in range(T):
            # 0.5 * (xu - ^xu)^T l_xuxu (xu-^xu) + l_xu (xu-^xu) + l_c
            _xu = np.concatenate(xu[t], axis=0)
            l_const[t] += 0.5 * _xu.T.dot(l_xuxu[t]).dot(_xu) - l_xu[t].dot(_xu)
            l_xu[t] -= _xu.T.dot(l_xuxu[t])

        cost_new_new = LQGeval(policy, dynamics, l_xuxu, l_xu, False, l_const)
        if cost_old_old is not None:
            epsilon = adjust_epsilon_rule(epsilon, cost_old_old, cost_new_old, cost_new_new)
            epsilon = max(min(epsilon, max_epsilon_ratio *_epsilon), min_epsilon_ratio * _epsilon)

        #fake_policy = [LinearGaussian(np.zeros((dU, dX)), xu[i][1]) for i in range(T)]
        #print(LQGeval(fake_policy, dynamics, l_xuxu, l_xu, False, l_const, deterministic=1))
        #print(cost)
        old = policy
        policy, eta = KL_LQG(dynamics, l_xuxu, l_xu, policy, epsilon=epsilon)
        print('kl', kl_divergence(policy, old, dynamics))

        cost_new_old = LQGeval(policy, dynamics, l_xuxu, l_xu, False, l_const)
        cost_old_old = cost_new_new

    return xu, cost
