import numpy as np
import torch

class XX:
    def __array__(self)-> np.ndarray:
        return np.array([1, 2])

    def __iter__(self):
        return self.__array__()

a = XX()
print(np.array(a))
