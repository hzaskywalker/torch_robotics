import torch
import nn

class Normalizer(nn.Module):
    # torch version
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range
        # some local information
        self.sum = nn.Paramter(torch.zeros(self.size), requires_grad=False)
        self.sumsq = nn.Paramter(torch.zeros(self.size), requires_grad=False)
        self.count = nn.Paramter(torch.zeros(1), requires_grad=False)

        # get the mean and std
        self.mean = nn.Paramter(torch.zeros(self.size), requires_grad=False)
        self.std = nn.Paramter(torch.ones(self.size), requires_grad=False)

    # update the parameters of the normalizer
    def update(self, v: torch.Tensor):
        v = v.reshape(-1, *self.size)
        # do the computing

        self.sum += v.sum(dim=0)
        self.sumsq += (v**2).sum(dim=0)
        self.count[0] += v.shape[0]

        self.recompute_stats()


    def recompute_stats(self):
        # calculate the new mean and std
        self.mean[:] = self.sum / self.count
        std = ((self.sumsq/self.count) - self.mean **2)
        self.std[:] = std.clamp(self.eps**2, float('inf'))

    # normalize the observation
    def normalize(self, v: torch.Tensor, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        return ((v - self.mean)/self.std).clamp(-clip_range, clip_range)

    def __call__(self, *args, **kwargs):
        return self.normalize(*args, **kwargs)

    def denormalize(self, v):
        return v * self.std + self.mean
