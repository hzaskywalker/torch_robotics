import torch
import inspect
import os
import numpy as np

class as_input:
    def __init__(self, dim=2, device='cuda:0', batch_dim=0):
        self.dim = dim
        self.device = device
        self.batch_dim = batch_dim

    def __call__(self, f):
        self.start = 0
        args = inspect.getfullargspec(f).args
        if len(args) > 0:
            if args[0] == 'self':
                self.start = 1

        def wrapped_f(*args):
            is_single = len(args[self.start].shape) < self.dim
            is_np = isinstance(args[self.start], np.ndarray)

            if is_np:
                args = args[:self.start] + tuple(torch.Tensor(i).to(self.device) for i in args[self.start:])
            if is_single:
                args = args[:self.start] + tuple(i.unsqueeze(0) for i in args[self.start:])
            out = f(*args)
            if not isinstance(out, tuple):
                if is_np:
                    out = out.detach().cpu().numpy()
                if is_single:
                    out = out[0]
            else:
                if is_np:
                    out = tuple(i.detach().cpu().numpy() for i in out)
                if is_single:
                    out = tuple(i.squeeze(self.batch_dim) for i in out)
            return out
        return wrapped_f

class batch_runner:
    def __init__(self, batch_size=128, show=False):
        self.batch_size = batch_size
        self.show = show

    def __call__(self, f):
        def wrapped_f(*args):
            N = len(args[0])
            l = 0
            outs = []
            while l < N:
                r = min(N, l+self.batch_size)
                if self.show:
                    print(l, r, '...', N)

                inp = [i[l:r] for i in args]
                outs.append(f(*inp))
                l = r
            return torch.cat(outs, dim=0)

        return wrapped_f

class cache:
    def __init__(self, path):
        self.path = path

    def __call__(self, f):
        def wrapped_f(*args, **kwargs):
            if self.path is not None and os.path.exists(self.path):
                print('loading..', self.path)
                return torch.load(self.path)
            out = f(*args, **kwargs)
            if self.path is not None:
                torch.save(out, self.path)
            return out
        return wrapped_f
