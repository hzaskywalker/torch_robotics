import numpy as np


class Parameter:
    def __init__(self, data):
        self.data = data

    def forward(self, sim):
        raise NotImplementedError

    @property
    def size(self):
        return int(np.prod(self.data.shape))

    @property
    def shape(self):
        return self.data.shape

    def update(self, data):
        self.data = data.copy().reshape(
            self.data.shape)
