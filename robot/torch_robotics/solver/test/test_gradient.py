import torch
from robot import tr
from robot.torch_robotics.solver.lcp import SlowLemkeAlgorithm, ProjectedGaussSiedelLCPSolver, CvxpySolver, lemke
from robot.utils import Timer

def test_lemke_gradient():
    """
    M = tr.togpu([
        [[1, 0, 1],
        [0, 1, 0],
        [1, 1, 0]],
    ])
    """
    torch.manual_seed(0)

    #M = tr.togpu([
    #    [[1, 0.5, 0],
    #     [0.5, 1, 0.3],
    #     [0, 0.3, 1]],
    #]).expand(128, -1, -1)
    n = 30
    batch_size = 2
    #batch_size = 1
    L = torch.randn((batch_size, n, n))
    M = tr.dot(L, tr.transpose(L)) + torch.eye(n)[None,:] * 0.0001

    q = torch.randn((batch_size, n))

    M.requires_grad = True
    q.requires_grad = True

    solver = {
        'lemke': SlowLemkeAlgorithm(niters=10000),
        'lemke2': lemke,
        'cvxpy': CvxpySolver(n),
        'pgs': ProjectedGaussSiedelLCPSolver(niters=100),
    }

    sols = []
    M_grads = []
    q_grads = []
    for name in ['lemke', 'cvxpy', 'lemke2', 'pgs']:
        M.grad, q.grad=None, None
        with Timer(name+' time'):
            with Timer(name + ' forward'):
                out = solver[name](M, q)
            with Timer(name + ' backward'):
                (out ** 2).sum().backward()
        M_grads.append(M.grad)
        q_grads.append(q.grad)
        sols.append(out)
    sols = torch.stack(sols, dim=0)
    M_grads = torch.stack(M_grads, dim=0)
    q_grads = torch.stack(q_grads, dim=0)

    print((sols[0]-sols[1]).abs().max())
    print((M_grads[0]-M_grads[1]).abs().max())
    print((M_grads[2]-M_grads[1]).abs().max())
    print((M_grads[3]-M_grads[1]).abs().max())
    print((M_grads[3]-M_grads[2]).abs().max())



if __name__ == '__main__':
    test_lemke_gradient()