import random
import torch
import numpy as np


def seed(s, env=None):
    if env is not None:
        env.seed(s)
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
