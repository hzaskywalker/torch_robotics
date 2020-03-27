from .device import MI, MO, device
from .utils import batched_index_select, write_video
from .tags import *
from .seed import seed
from .set_logger import set_logger, info
from .timer import Timer
from .tensorboard import Visualizer
from .pca import pca
from .arm_utils import play
from .fps import farthest_point_sample
from .soft_update import soft_update
from .draw import *
from .mp import mp_run
from . import models
from .shortest_path import *
from .packed_tensor import *
from .distributions import LinearGaussian
from .control_utils import *
from .decorators import *

from .trainer import merge_training_output, AgentBase, train_loop, add_parser
from .rl_utils import *

import torch
import numpy as np
def togpu(x, dtype=torch.float32):
    if x is None:
        return x
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, device='cuda:0', dtype=dtype)
    else:
        if torch.cuda.is_available():
            x = x.cuda()
    return x

def tocpu(x):
    if x is None:
        return x
    if isinstance(x, list) or isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()
