import torch
from robot import tr
from robot.torch_robotics.solver.lcp import SlowLemkeAlgorithm, ProjectedGaussSiedelLCPSolver, CvxpySolver, lemke, QpthSolver, LCPPhysics
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

    n = 30 # only if n<=30, lemke is faster than interior point methods..
    # it seems that interior methods are better in any sense ...
    # as there are very good implementation

    batch_size = 512

    L = torch.randn((batch_size, n, n))
    M = tr.dot(L, tr.transpose(L)) + torch.eye(n)[None,:] * 0.001

    q = torch.randn((batch_size, n))

    M = M.cuda()
    q = q.cuda()

    M.requires_grad = True
    q.requires_grad = True

    solver = {
        'lemke': SlowLemkeAlgorithm(niters=10000),
        'lemke2': lemke,
        'cvxpy': CvxpySolver(n),
        'pgs': ProjectedGaussSiedelLCPSolver(niters=50),
        'qpth': QpthSolver(),
        'lcp_phys': LCPPhysics()
    }

    sols = []
    M_grads = []
    q_grads = []

    out = solver['lemke'](M, q) # initialize

    algo = ['lemke2', 'lcp_phys', 'lemke', 'pgs', 'cvxpy', 'qpth']
    for name in algo:
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

    diff_sol = (sols[:, None] - sols[None, :]).reshape(len(algo), len(algo),
                                                         -1).abs().max(dim=-1)[0].detach().cpu().numpy()
    from robot.utils.print_table import print_table
    print_table([['']+algo]+[[name] + i.tolist() for name,i in zip(algo, diff_sol)])

    diff = (M_grads[:, None] - M_grads[None, :]).reshape(len(algo), len(algo),
                                                         -1).abs().max(dim=-1)[0].detach().cpu().numpy()
    from robot.utils.print_table import print_table
    print_table([['']+algo]+[[name] + i.tolist() for name,i in zip(algo, diff)])
    print(M_grads[1].max())



if __name__ == '__main__':
    test_lemke_gradient()