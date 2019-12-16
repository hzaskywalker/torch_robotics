import torch

def coor2d(n, m=None, device='cuda:0'):
    if m is None:
        m = n
    index_s = torch.arange(n).long().to(device)
    index_t = torch.arange(m).long().to(device)
    s = index_s[:, None].expand(n, m).contiguous()
    t = index_t[None, :].expand(n, m).contiguous()
    return torch.stack((t, s), dim=2)
