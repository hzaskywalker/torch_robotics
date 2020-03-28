import torch
from torch import nn
import os
import cv2
import tqdm
import gym
import numpy as np


def batch_gen(batch_size, *args):
    assert type(batch_size) == int
    length = len(args[0])
    l = 0
    while l < length:
        r = min(l + batch_size, length)
        if len(args) == 1:
            yield args[0][l:r]
        else:
            yield [i[l:r] for i in args]
        l = r


def batched_index_select(input, dim, index):
    assert len(index.shape) <= 2
    views = [input.shape[0]] + \
        [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)


def write_video(gen, path=None):
    out = None
    for img in gen:
        if path is not None:
            if out is None:
                out = cv2.VideoWriter(
                    path, cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (img.shape[1], img.shape[0]))
            out.write(img)
        else:
            cv2.imshow('x', img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    if out is not None:
        out.release()
