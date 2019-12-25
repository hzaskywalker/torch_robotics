import torch
import numpy as np

def norm(a, p=2, dim=-1, eps=1e-14):
    #return a.renorm(p=p, dim=dim, maxnorm=eps)/eps
    return a/a.norm(p=p, dim=dim).clamp(eps, float('inf'))

def rmat(a: torch.Tensor, row=3):
    assert a.shape[-1] == 6
    a1 = a[..., 0::2]
    a2 = a[..., 1::2]

    b1 = norm(a1)
    b2 = norm(a2 - (a2 * b1).sum(dim=-1, keepdim=True) * b1)
    if row == 3:
        b3 = torch.cross(b1, b2)
        return torch.stack((b1, b2, b3), dim=-1)
    else:
        return torch.stack((b1, b2), dim=-1)

def rmul(a: torch.Tensor, b: torch.Tensor):
    #return mata @ rmat(b, row=2)
    return torch.matmul(rmat(a), rmat(b,row=2)).flatten(start_dim=-2)

def rdist(predict, target):
    R = rmat(predict)
    R_gt = rmat(target)

    i0 = torch.eye(3, device=target.device)
    #i2 = torch.diag(torch.Tensor([1, -1, -1], device=target.device))
    i1 = torch.FloatTensor(np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ])).to(target.device)
    a = (R-torch.matmul(R_gt, i0))**2
    b = (R-torch.matmul(R_gt, i1))**2
    return torch.min(a.sum(dim=(-2, -1)), b.sum(dim=(-2, -1)))


def inv(a):
    d = a.dim()
    return rmat(a).permute(*list(range(d-1)), d, d-1).contiguous()[..., :2].flatten(-2)


if __name__ == '__main__':
    a = torch.Tensor(np.array([1, 0, 0, 1, 0, 0]))
    b = torch.Tensor(np.array([1, -1, 2, -1, 3, 1]))

    #print(rdist(a, b))
    print(rmul(rmul(a, b), inv(b)))

