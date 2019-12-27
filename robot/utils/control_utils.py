import gym
import tqdm
import numpy as np


def evaluate(env: gym.Env, controller, timestep=200, num_episode=10):
    ans = []
    for _ in tqdm.trange(num_episode):
        state = env.reset()
        try:
            controller.reset()  # reset the controller if it has
        except AttributeError:
            pass

        total = 0
        for j in tqdm.trange(timestep):
            action = controller(state)
            state, r, d, _ = env.step(action)
            total += r
            if d:
                break
        ans.append(total)
    return np.mean(ans)


def rollout(env, controller, x=None, timestep=200):
    if x is not None:
        x = env.reset(x)
    else:
        x = env.reset()

    try:
        controller.reset() # reset the controller if it has
    except AttributeError:
        pass

    xs, us = [], []
    for i in range(timestep):
        u = controller(x)
        xs.append(x)
        us.append(u)
        x = env.step(u)[0]

    xs.append(x)
    us.append(u)
    return np.array(xs), np.array(us)
