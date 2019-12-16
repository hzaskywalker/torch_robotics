# farthest point sampling based on l2
import numpy as np
from numpy import random
import torch


def farthest_point_sample(data, K=1000, basis=None, eps=1e-3, inf=100000):
    input_np = 0
    if isinstance(data, np.ndarray):
        data = torch.Tensor(data).cuda()
        input_np = 1

    data_ = data.view(len(data), -1)
    dist = torch.zeros(len(data),).to(data.device) + inf

    if basis is not None:
        basis = basis.view(len(basis), -1)
        new_dist = ((data_[:, None, :] - basis[None, :])
                    ** 2).mean(dim=2).min(dim=1)[0]
        dist = torch.stack((dist, new_dist)).min(dim=0)[0]

    choosed = []
    while len(choosed) < K:
        if dist.max() < eps:
            break
        idx = dist.argmax()
        new = data[idx]
        choosed.append(idx)
        new_dist = ((data_ - new.view(-1)[None, :])**2).mean(dim=1)
        dist = torch.stack((dist, new_dist)).min(dim=0)[0]
    if len(choosed) == 0:
        return []
    #print('Found {} points'.format(len(choosed)))
    choosed = torch.stack(choosed)
    if input_np:
        choosed = choosed.detach().cpu()
    return choosed
