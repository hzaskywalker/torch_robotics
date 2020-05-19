import torch
from robot import tr
from robot.torch_robotics.solver.slow import Solver

def main():
    solver = Solver()

    P = tr.togpu([[
        [1, 0.1],
        [0.1, 1]
    ]])
    q = tr.togpu([[
        -1, -1,
    ]])
    G = tr.togpu([[
        [1, 0.2],
        [0, 1]
    ]])

    h = tr.togpu([[
        -1, -1
    ]])

    sol = solver(P, q, G, h, 2, 0, 0)
    print(sol)

if __name__ == '__main__':
    main()