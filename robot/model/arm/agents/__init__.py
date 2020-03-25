from .simple_rollout import RolloutAgent

def make_agent(model, info, args):
    # max_a is very important to make it work...
    if not args.resume:
        agent = RolloutAgent(model, lr=args.lr, compute_reward=info.compute_reward, encode_obs=info.encode_obs,
                             max_a=info.max_a, max_q=info.max_q, max_dq=info.max_dq)
    else:
        import torch, os
        agent = torch.load(os.path.join(args.path, 'agent'))
    return agent
