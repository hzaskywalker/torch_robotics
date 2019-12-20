import numpy as np
import tqdm
from robot.controller import gps2

def mc_kl(mu, sigma, dist, num_iter):
    p1 = gps2.LinearGaussian(np.zeros((mu.shape[0], mu.shape[0])), mu, sigma)
    ans = 0
    for i in tqdm.trange(num_iter):
        x = p1.sample(np.zeros(mu.shape[0]))
        x, u = x[:dist.dX], x[dist.dX:]
        ans += dist.logp(x, u)
    return ans/num_iter


def test_kl():
    W = np.array(
        [
            [3, 1],
            [1, 2],
        ]
    )
    k = np.array(
        [2, 1]
    )
    sigma = np.array([
        [1, 1.5],
        [0, 1],
    ])
    sigma = sigma.T.dot(sigma)

    dist = gps2.LinearGaussian(W, k, sigma)


    mu = np.array([
        0, 0, 0 ,0
    ])
    sigma = np.array([
        [1, 0, 0, 0],
        [0, 1, 1, 2],
        [0, 1, 1, 0],
        [0, 0, 0, 1],
    ])
    sigma = sigma.T.dot(sigma)

    print(dist.Elogp(mu, sigma))
    print(mc_kl(mu, sigma, dist, 10000000))


if __name__ == '__main__':
    test_kl()