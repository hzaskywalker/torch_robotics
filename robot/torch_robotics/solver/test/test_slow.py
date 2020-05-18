import torch
from robot import tr
from robot.torch_robotics.solver.slow import Solver

def main():
    solver = Solver()

    P = tr.togpu([[
        [1, 0],
        [0, 1]
    ]])
    q = tr.togpu([[
        -1, -1,
    ]])
    G = tr.togpu([[
        [1, 0],
        [0, 1]
    ]])

    h = tr.togpu([[
        0, 0
    ]])

    solver(P, q, G, h, 2, 0, 0)

if __name__ == '__main__':
    main()