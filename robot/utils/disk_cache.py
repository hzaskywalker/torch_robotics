import os
import torch

class Cache:
    def __init__(self, path):
        self.path = path

    def __call__(self, f):
        def wrapped_f(*args, **kwargs):
            if os.path.exists(self.path):
                print('loading..', self.path)
                return torch.load(self.path)
            out = f(*args, **kwargs)
            torch.save(out, self.path)
            return out
        return wrapped_f