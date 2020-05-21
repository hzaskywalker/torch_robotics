import torch
import numpy as np
from robot import tr
from robot.torch_robotics.solver.ipm import Solver, ConeQP

solver = Solver()
solver2 = ConeQP()

def check(a, b):
    assert ((a-b).abs().max() < 1e-6), f"{a} {b}"

def test_nonnegative_orthant():

    n, m = 5, 4

    A = np.random.randn(n, n)
    P = tr.togpu([A@A.T])
    q = tr.togpu([np.random.randn(n,)])
    G = tr.togpu([np.random.randn(m, n)])

    h = tr.togpu([np.random.randn(m,)])

    sol1 = solver(P, q, G, h, m, 0, 0)
    sol2 = solver2(P, q, G, h, m, 0, 0)
    check(sol1, sol2)


def test_second_order_cone():
    while True:
        n = np.random.randint(2, 5)
        n_Q, dim_Q = np.random.randint(1, n), np.random.randint(1, n)
        m = n_Q * dim_Q

        A = np.random.randn(n, n) + np.eye(n) * 0.001
        P = tr.togpu([A@A.T])
        q = tr.togpu([np.random.randn(n,)])
        G = tr.togpu([np.random.randn(m, n)])
        h = tr.togpu([np.random.randn(m,)])

        sol = solver(P, q, G, h, 0, n_Q, dim_Q)
        sol2 = solver2(P, q, G, h, 0, n_Q, dim_Q)
        check(sol, sol2)


if __name__ == '__main__':
    #test_nonnegative_orthant()
    test_second_order_cone()