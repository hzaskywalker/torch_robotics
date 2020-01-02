import os
from tensorboardX import SummaryWriter
import numpy as np
from types import GeneratorType


def read_tensorboard(dirs, key='is_success'):
    import tensorflow as tf
    import glob

    x = []
    y = []
    for f in glob.glob(dirs + '/*.tfevents*'):
        tt = 0
        for event in tf.train.summary_iterator(f):
            #print('one summary', event.step)
            for value in event.summary.value:
                if key in value.tag:
                    if value.HasField('simple_value'):
                        #print('value', value.simple_value)
                        x.append(event.step)
                        y.append(value.simple_value)
    return [x, y]


class TensorboardHelper:
    def __init__(self, filename, mode='train'):
        self.filename = filename
        self.tb_step_file_name = os.path.join(self.filename, 'tb_steps')
        self.writer = SummaryWriter(filename)
        self.init(0, mode)

        if os.path.exists(self.tb_step_file_name):
            with open(self.tb_step_file_name, 'r') as f:
                self.step = int(f.readline())

    def init(self, step, mode):
        self.step = step
        self.mode = mode

    def name(self, name):
        if self.mode is not None:
            return '{}_{}'.format(self.mode, name)
        else:
            return name

    def add_scalar(self, tag, value):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, self.step)

    def add_scalars(self, tag, values):
        """Log a scalar variable."""
        self.writer.add_scalars(tag, values, self.step)

    def add_image(self, tag, images):
        """Log a list of images."""

        img_summaries = []

        if images.shape[1] <= 3:
            images = images.transpose(0, 2, 3, 1)
        for i, img in enumerate(images):
            if img.shape[2] == 1:
                import cv2
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            self.writer.add_image(self.name('%s/%d'%(tag, i)), img.transpose(2, 0, 1), self.step)

    def add_embedding(self, tag, embed):
        mat, metadata, image = embed
        self.writer.add_embedding(mat=mat, metadata=metadata, label_img=image, global_step=self.step, tag=tag)

    def tick(self):
        self.step += 1

    def __del__(self):
        with open(self.tb_step_file_name, 'w') as f:
            f.write('{}'.format(self.step))

class Visualizer(object):
    def __init__(self, exp_path, num_iters=None, write_file=True):
        print('log path:', exp_path)
        self.path = exp_path
        self.tb = TensorboardHelper(exp_path)
        self.file = open(os.path.join(exp_path, 'log'), 'a+') if write_file else None
        if write_file:
            from .tags import get_tag
            self.file.write(get_tag()+'\n\n')
            self.file.flush()

    def dfs(self, prefix, outputs):
        if outputs is None:
            return
        if isinstance(outputs, dict):
            for i in outputs:
                self.dfs('{}/{}'.format(prefix, i), outputs[i])
        else:
            if not isinstance(outputs, tuple):
                if not isinstance(outputs, GeneratorType):
                    outputs = np.array(outputs)
                    if len(outputs.shape) == 4:
                        #print(prefix, outputs.shape)
                        self.tb.add_image(prefix, outputs)
                    else:
                        self.tb.add_scalar(prefix, outputs)
                        if self.file is not None:
                            self.file.write(str(self.tb.step) + ' ' + prefix + ' ' + str(outputs) + '\n')
                else:
                    from .utils import write_video
                    write_video(outputs, os.path.join(self.path, prefix.replace('/', '_'))+'.avi')
            else:
                self.tb.add_embedding(prefix, outputs)

    def __call__(self, outputs, step=None):
        if step is not None:
            self.tb.step = step
        self.dfs('output', outputs)
        if self.file is not None:
            self.file.flush()
        self.tb.tick()