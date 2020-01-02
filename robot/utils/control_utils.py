import gym
import tqdm
import numpy as np


def evaluate(env: gym.Env, controller, timestep=200, num_episode=10, use_tqdm=False):
    ans = []
    for _ in tqdm.trange(num_episode):
        state = env.reset()
        try:
            controller.reset()  # reset the controller if it has
        except AttributeError:
            pass

        total = 0
        ran = range if not use_tqdm else tqdm.trange
        for j in ran(timestep):
            action = controller(state)
            state, r, d, _ = env.step(action)
            total += r
            if d:
                break
        ans.append(total)
    return np.mean(ans)


def rollout(env, controller=None, x=None, timestep=200):
    if x is not None:
        x = env.reset(x)
    else:
        x = env.reset()

    try:
        controller.reset() # reset the controller if it has
    except AttributeError:
        pass

    if controller is None:
        def random_policy(*args, **kwargs):
            return env.action_space.sample()
        controller = random_policy

    xs, us = [], []
    for i in range(timestep):
        u = controller(x)
        xs.append(x)
        us.append(u)
        x = env.step(u)[0]

    xs.append(x)
    us.append(u)
    return np.array(xs), np.array(us)
