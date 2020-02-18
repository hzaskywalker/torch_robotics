import numpy as np

class LeastSqaure:
    def __init__(self, n, m, std_A=1., std_b=1.):
        # m constraints.
        # n outputs
        self.n = n
        self.m = m
        self.std_A = std_A
        self.std_b = std_b

    def reset(self, scene=None):
        if scene is not None:
            self.A, self.b = scene
        else:
            self.A = np.random.normal(size=(self.n, self.m)) * self.std_A
            self.b = np.random.normal(size=(self.m,)) * self.std_b

        return self.A, self.b

    def step(self, x):
        diff = x @ self.A - self.b
        return {
            't': diff,
            'reward': (diff**2).sum(axis=-1),
            'scene': (self.A, self.b)
        }
