from .device import MI, MO, device
from .utils import batched_index_select, save, resume, resume_if_exists, dict2field
from .tags import *
from .seed import seed
from .set_logger import set_logger, info
from .timer import Timer
from .disk_cache import Cache
from .tensorboard import Visualizer
from .pca import pca
from .arm_utils import play
from .fps import farthest_point_sample
from .soft_update import soft_update
from .draw import *
from .mp import mp_run
from .trainer import merge_training_output, AgentBase, train_loop, add_parser
from . import models
from .shortest_path import *
from .packed_tensor import *
from .distributions import LinearGaussian

import torch
import numpy as np
def togpu(x):
    x = torch.Tensor(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def tocpu(x):
    if isinstance(x, list) or isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()
