import torch
import numpy as np

def padding(data, length):
    dim = -1
    mask = torch.arange(data.shape[length.dim()], device=data.device)

    shape = [1,] * (len(length.shape) + 1)
    shape[dim] = -1
    mask = mask.reshape(*shape)
    mask = mask < length.unsqueeze(dim)

    mask = mask.reshape(
        *mask.shape, *((1,) * (len(data.shape) - len(mask.shape))))
    return data * mask.float()
