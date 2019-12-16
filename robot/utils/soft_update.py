import torch

def soft_update(target, source, tau):
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.copy_(
                target_param * (1.0 - tau) + param.detach() * tau
            )