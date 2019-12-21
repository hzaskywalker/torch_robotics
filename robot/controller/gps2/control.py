# Control, given an environment that support reset
import copy
import numpy as np
import tqdm
from robot.controller.gps2 import LinearGaussian, KL_LQG
from robot.envs.arms.env import Env

def initial_policy(dX, dU, N):
    policy = []
    for i in range(N):
        policy.append(
            LinearGaussian(np.zeros((dU, dX)), np.zeros(dU), np.eye(dU) * 0.01)
        )
    return policy


def rollout(policy, env: Env, start, target, T):
    x = np.array(start)
    cost = 0
    xu = []
    for i in range(T):
        u = policy[i].sample(x)
        l, _, _ = env.cost(x, u)
        cost += l * env.dt
        xu.append([x, u])

        if i!= T-1:
            x = np.array(env.forward(x, u))

    final_cost = env.cost_final(x, target)[0]

    print('start', env.arm.position(start)[:, -1])
    print('reached', env.arm.position(x)[:, -1])
    print('target', target)

    cost += final_cost
    return xu, cost


def LQR_control(env, initial: LinearGaussian, target: np.ndarray, T, num_iters, epsilon=1.):
    # hope to reach the target from the initial position to the target position.
    dX, dU = env.dof * 2, env.dof
    policy = initial_policy(dX, dU, T)

    for i in tqdm.trange(num_iters + 1):
        # rollout
        xu, cost = rollout(policy, env, initial.sample(), target, T)
        if i == num_iters:
            break
        print(i, cost)

        dynamics = [copy.deepcopy(initial)]
        l_xuxu = []
        l_xu = []
        for t in range(T-1):
            # build new dynamics
            x_t, u_t = xu[t]
            A, B = env.finite_differences(x_t, u_t)
            # x(t+1) = f(x(t), u(t)) = A(x -^x) + B(u-^u) + x_{t+1}
            dynamics.append(
                LinearGaussian(np.concatenate((A, B), axis=1), A.dot(-x_t) + B.dot(-u_t) + xu[t+1][0])
            )
            _, _l_xu , _l_xuxu = env.cost(x_t, u_t)
            l_xuxu.append(_l_xuxu * env.dt)
            l_xu.append(_l_xu * env.dt)

        _, _final_l_x, _final_l_xx = env.cost_final(xu[-1][0], target)
        l_xuxu += [_final_l_xx]
        l_xu += [_final_l_x]

        new_policy, eta = KL_LQG(dynamics, l_xuxu, l_xu, policy, epsilon=epsilon)

        # Note we need to calculate optimal change to LQG return policy as g(x) = K (u-^u) + k (x-^x) + k
        for t in range(T):
            x_t, u_t = xu[t]
            new_policy[t].b = u_t + new_policy[t].b - new_policy[t].W.dot(x_t)

    return xu, cost
