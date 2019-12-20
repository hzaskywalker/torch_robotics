# various cost functions and testing functions.
"""
d x^T A x = x^T(A+A^T)
          = 2x^T A if A is symmetry

d_xx x^TAx = A + A^T or 2A if A is symmetry
"""
import numpy as np

def cost_fk(dX, dU, u, type='x'):
    assert type in 'xu'
    # 1/2(x-u)**2 = 1/2x**2 - xu + 1/2u^2
    # l_xx = 1
    # l_x = u
    l_uu = np.zeros((dX + dU, dX + dU))
    l_u = np.zeros((dX + dU))

    idx = slice(dX) if type == 'x' else slice(dX, dX + dU)

    l_uu[idx, idx] = np.eye(dX if type == 'x' else dU)
    l_u[idx] = -np.array(u)
    return l_uu, l_u
