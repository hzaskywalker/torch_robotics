import gym
import tqdm
import numpy as np


def evaluate(env: gym.Env, controller, timestep=200, num_episode=10, use_tqdm=False, print_reward=False):
    ans = []
    for _ in tqdm.trange(num_episode):
        state = env.reset()
        if 'reset' in controller.__dir__():
            controller.reset()  # reset the controller if it has

        total = 0
        ran = range if not use_tqdm else tqdm.trange
        for j in ran(timestep):
            action = controller(state)
            state, r, d, _ = env.step(action)
            total += r
            if d:
                break
        ans.append(total)
        if print_reward:
            print(ans[-1])

    return np.mean(ans)


def rollout(env, controller=None, x=None, timestep=200, use_tqdm=False):
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

    xs, us, rs = [], [], []
    ran = tqdm.trange if use_tqdm else range
    for _ in ran(timestep):
        u = controller(x)
        xs.append(x)
        us.append(u)
        x, r = env.step(u)[:2]

        rs.append(r)

    xs.append(x)
    us.append(u)
    return np.array(xs), np.array(us), np.array(rs)
