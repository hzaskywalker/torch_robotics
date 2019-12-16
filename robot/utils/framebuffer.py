import numpy as np
from collections import deque
from mp.utils import togpu

class NaiveBuffer:
    ## padding the points if the scene is too large
    def __init__(self, batch_size, maxlen):
        self.buffer = deque(maxlen=maxlen) #TODO: lazy to store the training scene
        self.batch_size = batch_size

    def add(self, *args):
        self.buffer.append(args)

    def sample(self):
        output = []
        for i in range(self.batch_size):
            output.append(self.buffer[np.random.choice(len(self.buffer))])
        output = np.array(output).T
        return [togpu(i.tolist()) for i in output]