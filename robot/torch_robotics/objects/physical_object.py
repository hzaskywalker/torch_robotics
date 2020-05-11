# basic objects
import torch

class PhysicalObject:
    def iter(self):
        raise NotImplementedError

    @classmethod
    def new(cls, *iters):
        return cls(*iters)

    @classmethod
    def stack(cls, arrays, dim=0):
        outs = []
        arrays = [i.iter() for i in arrays]
        for j in range(len(arrays[0])):
            if arrays[0][j] is not None:
                outs.append(torch.stack([i[j] for i in arrays], dim=dim))
            else:
                outs.append(None)
        return cls.new(*outs)

    @classmethod
    def cat(cls, arrays, dim=0):
        outs = []
        arrays = [i.iter() for i in arrays]
        for j in range(len(arrays[0])):
            if arrays[0][j] is not None:
                outs.append(torch.cat([i[j] for i in arrays], dim=dim))
            else:
                outs.append(None)
        return cls.new(*outs)

    def __getitem__(self, item):
        return self.new(*[i[item] for i in self.iter()])

    def __setitem__(self, key, value):
        for a, b in zip(self.iter, value.iter()):
            a[key] = b

    def __len__(self):
        return self.iter()[0].shape[0]

    def fk(self):
        raise NotImplementedError(f"fk for {self.__class__} is not implemented")
