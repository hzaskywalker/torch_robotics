import numpy as np
import torch

class XX:
    def __array__(self)-> np.ndarray:
        return np.array([1, 2])

    def __iter__(self):
        return self.__array__()

a = XX()
print(np.array(a))

class XX:
    @classmethod
    def add(cls, a, b):
        return cls.add2(a, b)

    @classmethod
    def add2(cls, a, b):
        return a+b

b = XX()
print(b.add(1, 2))
