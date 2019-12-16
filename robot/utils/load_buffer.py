import os
import torch

def load_buffer(path):
    from tools.dataset import EpisodeBuffer
    framebuffer_dict = torch.load(os.path.join(path, 'framebuffer'))
    framebuffer = EpisodeBuffer(1000000)
    framebuffer.load_dict(framebuffer_dict)
    return framebuffer
