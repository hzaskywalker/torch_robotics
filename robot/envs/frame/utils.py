import numpy as np
import torch
from collections import OrderedDict


TYPE_DICT = {
    np.dtype('float32'): torch.float,
    np.dtype('float64'): torch.float,
    np.dtype('int64'): torch.long,
}



def cat(out, dim):
    if isinstance(out[0], np.ndarray):
        return np.concatenate(out, axis=dim)
    else:
        return torch.cat(out, dim=dim)


def serialize(v, is_batch):
    if isinstance(v, np.ndarray) or isinstance(v, torch.Tensor):
        if is_batch:
            return v.reshape(v.shape[0], -1)
        else:
            return v.reshape(-1)
    else:
        return cat([serialize(v, is_batch) for _, v in v.items()], dim=-1)

def size(shape):
    if isinstance(shape, OrderedDict):
        return sum([size(i) for i in shape.values()])
    elif isinstance(shape, tuple) or isinstance(shape, list):
        return int(np.prod(shape))
    else:
        raise NotImplementedError
        return shape.size

def deserialize(v, shape, is_batch):
    if isinstance(v, np.ndarray) or isinstance(v, torch.Tensor):
        if is_batch:
            return v.reshape(v.shape[0], *shape)
        else:
            return v.reshape(*shape)
    elif isinstance(shape, OrderedDict):
        raise NotImplementedError
        l = 0
        out = OrderedDict()
        for i, spec in shape.items():
            s = size(spec)
            d = v[l:l + s] if not is_batch else v[:, l:l + s]
            out[i] = deserialize(d, spec, is_batch)
            l += s
        return out

def to_numpy(v):
    if isinstance(v, np.ndarray):
        return v
    elif isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy()
    elif isinstance(v, OrderedDict):
        return OrderedDict([(i, to_numpy(v))for i, v in v.items()])
    else:
        raise NotImplementedError


def to_tensor(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, np.ndarray):
        return torch.tensor(data, dtype=TYPE_DICT[data.dtype], device=device)
    elif isinstance(data, OrderedDict):
        return OrderedDict([(i, to_tensor(v, device)) for i, v in data.items()])
    else:
        raise NotImplementedError
