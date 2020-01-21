import numpy as np
import torch


TYPE_DICT = {
    np.dtype('float32'): torch.float,
    np.dtype('float64'): torch.float,
    np.dtype('int64'): torch.long,
}


def serialize(v, is_batch):
    if isinstance(v, np.ndarray) or isinstance(v, torch.Tensor):
        if is_batch:
            return v.reshape(v.shape[0], -1)
        else:
            return v.reshape(-1)
    else:
        return v.serialize()


def cat(out, dim):
    if isinstance(out[0], np.ndarray):
        return np.concatenate(out, axis=dim)
    else:
        return torch.cat(out, dim=dim)


def to_numpy(v):
    if isinstance(v, np.ndarray):
        pass
    elif isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    else:
        v = v.numpy()
    return v


def to_tensor(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, np.ndarray):
        return torch.tensor(data, dtype=TYPE_DICT[data.dtype])
    else:
        return data.tensor(device)
