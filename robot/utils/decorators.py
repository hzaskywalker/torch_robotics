import torch
import os
import numpy as np

class as_input:
    def __init__(self, dim=2):
        self.dim = dim

    def __call__(self, f):
        def wrapped_f(*args):
            is_single = len(args[0].shape) < self.dim
            is_np = isinstance(args[0], np.ndarray)

            if is_np:
                args = [torch.Tensor(i).cuda() for i in args]
            if is_single:
                args = [i[None,:] for i in args]
            out = f(*args)
            if is_np:
                out = out.detach().cpu().numpy()
            if is_single:
                out = out[0]
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
