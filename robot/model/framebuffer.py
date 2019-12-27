"""
What a framebuffer should support:
    1. init with random trajectories
"""
import torch
import numpy as np
from robot.utils import togpu

class CircleBuffer:
    def __init__(self, maxlen):
        self.buffer = [None] * maxlen
        self.maxlen = maxlen
        self.index = 0
        self.size = 0
        self.start = 0

    def append(self, obj):
        if self.index == self.start:
            self.popleft()

        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.maxlen)
        self.index = (self.index + 1) % self.maxlen

    def popleft(self):
        self.start = (self.start + 1) % self.maxlen

    def __getitem__(self, indices):
        if not isinstance(indices, int):
            return [self.buffer[index % self.maxlen] for index in indices]
        else:
            return self.buffer[(self.start + indices) % self.maxlen]

    def sample(self, batch_size):
        if self.index <= self.start:
            idx = np.random.choice(self.maxlen - self.start + self.index, batch_size) + self.start
        else:
            idx = np.random.choice(self.index - self.start, batch_size) + self.start
        return self.__getitem__(idx)


class TrajBuffer:
    def __init__(self, maxlen, valid_ratio=0.2, batch_size=200):
        self.maxlen = maxlen
        self.data = [CircleBuffer(self.maxlen) for i in range(2)]

        self.train = CircleBuffer(maxlen)
        self.valid = CircleBuffer(maxlen)
        self.cur = 0
        self.valid_ratio = valid_ratio
        self.batch_size = 200

    def update(self, s, a, *args):
        # everytime we will add a trajectory
        for it, i in enumerate([s, a] + list(args)):
            if it >= len(self.data):
                self.data.append(CircleBuffer(self.maxlen))
            buffer = self.data[it]
            for j in i:
                buffer.append(j)

        n = len(args[0])
        for it in range(n-1):
            if np.random.random() < self.valid_ratio:
                self.valid.append(self.cur)
            else:
                self.train.append(self.cur)
            self.cur += 1

        while self.train[0] < self.cur - self.maxlen:
            self.train.popleft()
        while self.valid[0] < self.cur - self.maxlen:
            self.valid.popleft()

    def save(self, path):
        if path is not None:
            print('saving...', path)
            torch.save([self.data, self.train, self.valid, self.cur, self.batch_size], path)

    def load(self, path):
        if path is not None:
            self.data, self.train, self.valid, self.cur, self.batch_size = torch.load(path)
            self.maxlen = self.train.maxlen

    def sample(self, mode='train', batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if mode == 'train':
            idx = self.train.sample(batch_size)
        else:
            idx = self.valid.sample(batch_size)

        s = self.data[0][idx]
        a = self.data[1][idx]
        t = self.data[2][idx + 1]
        out = [s, a, t] + [self.data[i][idx] for i in range(2, len(self.data))]
        return [togpu(i) for i in out]


def test():
    buffer = TrajBuffer(12, 0.2)


if __name__ == '__main__':
    test()
