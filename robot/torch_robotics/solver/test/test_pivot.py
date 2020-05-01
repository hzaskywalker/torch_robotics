import torch
import numpy as np
from robot import tr
from robot.torch_robotics.solver.lcp import SlowLemkeAlgorithm

def test():
    #M = torch.zeros()
    lemke = SlowLemkeAlgorithm()
    M = tr.togpu([
       [[1, -1.5, 0],
        [-1.5, 1, 0],
        [0, 0, 1]],
        [[2, -1, 0],
         [-1, 2, 0],
         [0, 0, 1]],
    ])
    M = lemke.create_MI1(M)
    print(M)

    variables = np.array([
        [1, 1, 0, 0, 0, 0, 0.1],
        [0, 0, 0, 0, 3, 1, 2],
    ])
    #print(M[1].detach().cpu().numpy()@variables[1])
    #exit(0)

    bas = []
    xs = []
    for i in variables:
        loc = np.where(i!=0)[0]
        bas.append(loc)
        xs.append(i[loc])

    bas = tr.togpu(bas).long()
    xs = tr.togpu(xs)
    entering = tr.togpu([2, 0]).long()

    q = tr.togpu([[0.4, 0.4, -0.1], [-2, 1, -1]])
    lemke.evaluate(M, bas, xs, q)

    new_bas, new_xs, leaving = lemke.pivot(M, bas, xs, entering)
    lemke.evaluate(M, new_bas, new_xs, q)


if __name__ == '__main__':
    test()