def test_coneqp():
    from robot.torch_robotics.solver.coneqp_python import coneqp
    from cvxopt import matrix, solvers

    A = matrix([[.3, -.4, -.2, -.4, 1.3],
                [.6, 1.2, -1.7, .3, -.3],
                [-.3, .0, .6, -1.2, -2.0]])
    b = matrix([1.5, .0, -1.2, -.7, .0])
    m, n = A.size

    I = matrix(0.0, (n, n))
    I[::n + 1] = 1.0

    G = matrix([-I, matrix(0.0, (1, n)), I])
    h = matrix(n * [0.0] + [1.0] + n * [0.0])
    dims = {'l': n, 'q': [n + 1], 's': []}
    x = solvers.coneqp(A.T * A, -A.T * b, G, h, dims)['x']
    print(x)

if __name__ == "__main__":
    test_coneqp()