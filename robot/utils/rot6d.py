import torch
import numpy as np

def norm(a: torch.Tensor, p=2, dim=-1, eps=1e-14):
    #return a.renorm(p=p, dim=dim, maxnorm=eps)/eps
    if isinstance(a, torch.Tensor):
        return a/a.norm(p=p, dim=dim, keepdim=True).clamp(eps, float('inf'))
    else:
        return a/np.linalg.norm(a, axis=dim, keepdims=True).clip(eps, float('inf'))

def rmat_np(a: np.ndarray, row=3):
    assert a.shape[-1] == 6
    a1 = a[..., 0::2]
    a2 = a[..., 1::2]

    b1: np.ndarray = norm(a1)
    b2: np.ndarray = norm(a2 - (a2 * b1).sum(axis=-1, keepdims=True) * b1)
    if row == 3:
        b3 = np.cross(b1, b2)
        return np.stack((b1, b2, b3), axis=-1)
    else:
        return np.stack((b1, b2), axis=-1)

def rmat(a: torch.Tensor, row=3):
    if isinstance(a, np.ndarray):
        return rmat_np(a, row)
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

def rmul_np(a: np.ndarray, b: np.ndarray):
    #return mata @ rmat(b, row=2)
    return np.matmul(rmat_np(a), rmat_np(b,row=2)).reshape(a.shape)

def rmul(a: torch.Tensor, b: torch.Tensor):
    if isinstance(a, np.ndarray):
        return rmul_np(a, b)
    #return mata @ rmat(b, row=2)
    return torch.matmul(rmat(a), rmat(b,row=2)).flatten(start_dim=-2)

def rdist(predict, target):
    P = rmat(predict, row=3)

    Q = rmat(target, row=3)
    d = Q.dim()
    Q = Q.permute(*list(range(d-2)), d-1, d-2).contiguous()

    diff = torch.matmul(P, Q)

    xx = 0.5 * (diff[..., 0, 0] + diff[..., 1, 1] + diff[..., 2, 2] - 1) #cos(theta)
    #return 1 - xx * xx
    return (xx - 1) ** 2


def r2quat(r: torch.Tensor):
    raise NotImplementedError
    # https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
    is_np = isinstance(r, np.ndarray)
    if is_np:
        r = torch.Tensor(r)
    if len(r.shape) == 2:
        out = torch.stack([r2quat(i) for i in r])
    else:
        m = rmat(r)
        if m[2, 2] < 0:
            if m[0, 0] > m[1, 1]:
                t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
                q = t, m[0, 1] + m[1, 0], m[2, 0] + m[0, 2], m[1, 2] - m[2, 1]
            else:
                t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
                q = m[0, 1] + m[1, 0], t, m[1, 2] + m[2, 1], m[2, 0] - m[0, 2]
        else:
            if m[0, 0] < -m[1, 1]:
                t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
                q = m[2, 0] + m[0, 2], m[1, 2] + m[2, 1], t, m[0, 1] - m[1, 0]
            else:
                # w form
                t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
                q = m[1, 2] - m[2, 1], m[2, 0] - m[0, 2], m[0, 1] - m[1, 0], t

        out = torch.stack((q[3], q[0], q[1], q[2]), dim=-1) * (0.5/torch.sqrt(t))

    if is_np:
        out = out.detach().numpy()
    return out

def inv(a):
    if isinstance(a, np.ndarray):
        d = len(a.shape)
        return rmat(a).transpose(*list(range(d-1)), d, d-1)[..., :2].reshape(a.shape)

    d = a.dim()
    return rmat(a).permute(*list(range(d-1)), d, d-1).contiguous()[..., :2].flatten(-2)


if __name__ == '__main__':
    a = np.array([1, 0, 0, 1, 0, 0])

    a = torch.Tensor(np.array([1, 0, 0, 1, 0, 0]))
    b = torch.Tensor(np.array([1, -1, 2, -1, 3, 1]))
    c = torch.Tensor(np.array([1, 0, 0, -1, 0, 0]))

    print(rdist(a, a))
    print(rdist(a, c))
    exit(0)
    #print(rmul(rmul(a, b), inv(b)))

