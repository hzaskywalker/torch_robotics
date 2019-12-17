# evaluation function to support zero-order controller
from robot.utils import togpu

def evaluate(env, state, actions):
    # Non-batched evaluation
    outs = []
    start = state.copy()
    for action in actions:
        total = 0
        state = start.copy()
        for a in action:
            state, r, done = env.forward(state, a)
            total += r
            if done:
                break
        outs.append(total)
    outs = togpu(outs)
    return outs
