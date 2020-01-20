import numpy as np
import torch


def to_numpy(v, is_batch):
    if isinstance(v, np.ndarray):
        pass
    elif isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    else:
        v = v.numpy()

    if is_batch:
        v = v.reshape(v.shape[0], -1)
    else:
        v = v.reshape(-1)
    return v


def to_tensor(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, np.ndarray):
        return torch.tensor(data, dtype=torch.float)
    else:
        return data.tensor(device)
