from .simple_rollout import RolloutAgent

def make_agent(model, cls_type, args):
    # max_a is very important to make it work...
    if not args.resume:
        agent = RolloutAgent(model, lr=args.lr, loss_weights={
            'q_loss': args.weight_q,
            'dq_loss': args.weight_dq,
            'ee_loss': args.weight_ee,
        })
    else:
        import torch, os
        agent = torch.load(os.path.join(args.path, 'agent'))
    return agent
