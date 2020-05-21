import torch
import numpy as np
from robot import tr
from robot.torch_robotics.solver.ipm import Solver, ConeQP

def check(a, b):
    assert ((a-b).abs().max() < 1e-6), f"{a} {b}"

def main():
    solver = Solver()
    solver2 = ConeQP()

    n, m = 5, 4

    A = np.random.randn(n, n)
    P = tr.togpu([A@A.T])
    q = tr.togpu([np.random.randn(n,)])
    G = tr.togpu([np.random.randn(m, n)])

    h = tr.togpu([np.random.randn(m,)])

    sol1 = solver(P, q, G, h, m, 0, 0)
    sol2 = solver2(P, q, G, h, m, 0, 0)
    check(sol1, sol2)

if __name__ == '__main__':
    main()