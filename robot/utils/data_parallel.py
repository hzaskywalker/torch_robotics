# A data parallel module
from inspect import isfunction
import torch
import numpy as np
import torch.multiprocessing as multip
from typing import List


class Worker(multip.Process):
    ASK = 1
    GET = 2
    EXIT = 3
    def __init__(self, cls, *args, **kwargs):
        multip.Process.__init__(self)
        self.cls = cls
        self.args = args
        self.kwargs = kwargs
        self.pipe, self.worker_pipe = multip.Pipe()
        self.start()

    def run(self):
        #env = self.create_fn[0](*self.create_fn[1:])
        if isfunction(self.cls):
            func = self.cls
        else:
            func = self.cls(*self.args, **self.kwargs)

        ans = None
        while True:
            op, data = self.worker_pipe.recv()
            if op == self.ASK:
                ans = func(*data)

            elif op == self.GET:
                self.worker_pipe.send(ans)

            elif op == self.EXIT:
                self.worker_pipe.close()
                return

    def ask(self, args):
        self.pipe.send([self.ASK, args])

    def get(self):
        self.pipe.send([self.GET, None])
        return self.pipe.recv()

    def close(self):
        self.pipe.send([self.EXIT, None])
        self.pipe.close()


class DataParallel:
    def __init__(self, n, cls, *args, **kwargs):
        self.n = n
        self.workers = []
        for i in range(n):
            self.workers.append(Worker(cls, *args, **kwargs))

    def __call__(self, *args):
        from robot.utils.utils import batch_gen
        n = min(len(args[0]), self.n)
        batch_size = (len(args[0]) + n-1)//n
        is_single_input = len(args) == 1
        for i, args in enumerate(batch_gen(batch_size, *args)):
            if is_single_input:
                self.workers[i].ask((args,))
            else:
                self.workers[i].ask(args)
        outs = None
        for i in range(n):
            o = self.workers[i].get()
            if isinstance(o, tuple):
                if outs is None:
                    outs = tuple([] for i in o)
                for x, y in zip(outs, o):
                    x.append(y)
            else:
                if outs is None:
                    outs = []
                outs.append(o)
        if isinstance(outs, tuple):
            outs = tuple(self.concat(i) for i in outs)
        else:
            outs = self.concat(outs)
        return outs

    def concat(self, x):
        if isinstance(x[0], torch.Tensor):
            return torch.cat(x, dim=0)
        elif isinstance(x[0], np.ndarray):
            return np.concatenate(x, axis=0)
        else:
            return sum(x, [])

    def __del__(self):
        for i in self.workers:
            i.close()

def test1():
    func = DataParallel(24, lambda x: (x+1, x+2))
    print(func(torch.arange(12)))

def test2():
    class Func:
        def __init__(self, d):
            self.d = d

        def __call__(self, x):
            return x + self.d
    func = DataParallel(24, Func, 12)
    print(func(torch.arange(12)))

if __name__ == '__main__':
    test1()
