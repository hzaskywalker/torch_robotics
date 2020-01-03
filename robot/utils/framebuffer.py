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
        if self.buffer[self.index] is not None and self.index == self.start:
            self.popleft()

        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.maxlen)
        self.index = (self.index + 1) % self.maxlen

    def popleft(self):
        self.buffer[self.start] = None
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

    def show(self):
        if self.index <= self.start:
            return self.buffer[self.start:] + self.buffer[:self.index]
        else:
            return self.buffer[self.start:self.index]


class TrajBuffer:
    def __init__(self, maxlen, valid_ratio=0.2, batch_size=200):
        self.maxlen = maxlen
        self.data = [CircleBuffer(maxlen) for i in range(2)]
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

        n = len(s)
        for it in range(n-1):
            if np.random.random() < self.valid_ratio:
                self.valid.append(self.cur)
            else:
                self.train.append(self.cur)
            self.cur += 1
        self.cur += 1

        while self.train[0] is not None and self.train[0] < self.cur - self.maxlen:
            self.train.popleft()

        while self.valid[0] is not None and self.valid[0] < self.cur - self.maxlen:
            self.valid.popleft()

    def save(self, path):
        if path is not None:
            print('saving...', path)
            torch.save([self.data, self.train, self.valid, self.cur, self.batch_size, self.valid_ratio], path)

    def load(self, path):
        if path is not None:
            self.data, self.train, self.valid, self.cur, self.batch_size, self.valid_ratio = torch.load(path)
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
        t = self.data[0][np.array(idx) + 1]
        try:
            out = [s, a, t] + [self.data[i][idx] for i in range(2, len(self.data))]
            return [togpu(np.array(i)) for i in out]
        except Exception as e:
            print(s)
            print(a)
            print(t)
            raise e

    def make_sampler(self, sample_method, mode, n):
        if sample_method == 'random':
            for i in range(n):
                yield self.sample(mode)
        else:
            for i in range(n):
                # num_epoch
                idxs = self.train.show()
                idxs = np.random.permutation(idxs)

                batch_size = self.batch_size
                num_batch = len(idxs) // batch_size

                for j in range(num_batch):
                    idx = idxs[j*batch_size:(j+1)*batch_size]
                    s = self.data[0][idx]
                    a = self.data[1][idx]
                    t = self.data[0+1][idx + 1]
                    yield togpu(s), togpu(a), togpu(t)



def test():
    buffer = TrajBuffer(100, 0.2)

    # duipai
    q = []
    todo = []
    for i in range(50):
        n = np.random.randint(3, 20)
        s = [np.random.random() for j in range(n)]
        buffer.update(s, s)
        for j in range(len(s)-1):
            todo.append(len(q) + j)
        q += s

    trainval = buffer.train.show() + buffer.valid.show()
    a = set(trainval)

    b = [j for j in todo if j >= len(q)-100]
    assert set(a) == set(b)

    for j in range(10):
        j = buffer.train.sample(128)

        a = np.array(buffer.data[0][j])
        b = np.array([q[k] for k in j])
        assert ((np.array(a) - np.array(b))**2).sum() < 1e-6



if __name__ == '__main__':
    test()
