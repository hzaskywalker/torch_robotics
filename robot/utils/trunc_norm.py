import numpy as np
import torch

normal = torch.distributions.normal.Normal(0, 1)

def parameterized_truncated_normal(size, device='cpu', a=-2, b=2):
    # tf implementation
    uniform = torch.rand(size=size, device=device)

    alpha = a
    beta = b

    alpha_normal_cdf = normal.cdf(alpha)
    p = alpha_normal_cdf + (normal.cdf(beta) - alpha_normal_cdf) * uniform

    #p = p.detach().cpu().numpy()
    #one = np.array(1, dtype=p.dtype)
    one = 1.
    epsilon = np.array(np.finfo(np.float32).eps, dtype=np.float32)
    #v = np.clip(2 * p - 1, -one + epsilon, one - epsilon)
    v = (2*p-1).clamp(-one + epsilon, one - epsilon)
    x = np.sqrt(2) * torch.erfinv(v) #torch.erfinv(torch.from_numpy(v))
    x = torch.clamp(x, a, b)

    return x

trunc_norm = parameterized_truncated_normal

def truncated_normal(size, device='cuda:0'):
    #tmp = tensor.new_empty(size + (4,)).normal_()
    tmp = torch.randn(size + (6,), device=device)
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    return tmp.gather(-1, ind).squeeze()


if __name__ == '__main__':
    # 500 * 30 * 4000
    import tqdm
    for i in tqdm.trange(1000):
        xx = parameterized_truncated_normal((500, 30, 400), device='cuda:0')
    print(xx.mean(), xx.std())
    print(xx.min(), xx.max())
    exit(0)
    x = 0
    for i in tqdm.trange(1000):
        cc = truncated_normal((500, 30, 804,))
        x = max(x, cc.max())
    print(x)

    print(cc.mean(), cc.std())
    print(cc.shape)
    print(cc.max(), cc.min())

    from scipy.stats import truncnorm
    x = truncnorm(-2, 2).rvs(size=(100000,))
    print(x.mean(), x.std())
