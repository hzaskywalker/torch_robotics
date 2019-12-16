# evaluation function to support zero-order controller
from robot.utils import togpu

def evalulate(env, state, actions, batch=True):
    # Non-batched evaluation
    if batch:
        assert len(state) == 1
        state = state[0]
        actions = actions[0]
    outs = []
    for action in actions:
        total = 0
        for a in action:
            state, r, done = env.forward(state, a)
            total += r
            if done:
                break
        outs.append(total)
    outs = togpu(outs)
    if batch:
        return outs[None,:]
    else:
        return outs
