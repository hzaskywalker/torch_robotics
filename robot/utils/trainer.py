## training utils
import torch
import os
import tqdm
import numpy as np


def add_parser(parser):
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--eval_episode', type=int, default=1000)
    parser.add_argument('--num_train_iters', type=int, default=200000)
    parser.add_argument('--num_valid_iters', type=int, default=100)
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--batch_norm', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    return parser


def calc_accuracy_recall(info, tp, fp, fn, tn):
    total = tp + fp + fn + tn
    info['accuracy'] = (tp + tn)/total
    info['recall'] = tp/(tp + fn + 1e-15)
    info['precision'] = tp/(tp + fp + 1e-15)
    info['num_positive_ratio'] = (tp + fn)/total
    return info

def merge_training_output(train_outputs):
    train_info = {i: np.mean([j[i] for j in train_outputs])
                  for i in train_outputs[0] if train_outputs[0][i].size == 1}

    if 'tp' in train_info:
        keys = ['tp', 'fp', 'fn', 'tn']
        calc_accuracy_recall(train_info, *[train_info[key] for key in keys])
        for key in keys:
            del train_info[key]

    return train_info


class AgentBase:
    def __init__(self, models, lr):
        self.models = models
        self.optim = torch.optim.Adam(models.parameters(), lr=lr)
        self.training = True

    def train(self):
        self.training = True
        self.models.train()
        return self

    def eval(self):
        self.training = False
        self.models.eval()
        return self

    def save(self, path, suffix=None):
        name = 'agent'
        if suffix is not None:
            name = f'agent_{suffix}'
        torch.save(self, os.path.join(path, name))

    @classmethod
    def load(cls, path):
        return torch.load(os.path.join(path, 'agent'))

    def cuda(self):
        if torch.cuda.is_available():
            self.models.cuda()
        return self

    def get_predict(self, points, pose):
        raise NotImplementedError

    def update(self, points, pose, labels):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        with torch.no_grad():
            predict = self.get_predict(*args, **kwargs)
        return predict


def train_loop(agent, dataset, path,
               num_iters, num_valid_iters, eval_episode,
               resume_path=None, train_vis_fn=[], valid_vis_fn=None, save_per_epsiode=False):
    """
    Example:
        def fn(train_info, train_outputs, data, output):
            train_info['xx'] = tocpu(output[-1])
    """
    assert resume_path is None

    agent_path = os.path.join(path, 'agent')
    if os.path.exists(agent_path):
        return torch.load(agent_path)

    from . import Visualizer, togpu
    train_vis = Visualizer(path)
    valid_vis = Visualizer(os.path.join(path, 'valid'))

    if valid_vis_fn is None:
        valid_vis_fn = train_vis_fn

    train_outputs = []
    agent.cuda()
    agent.train()
    for batch_id in tqdm.trange(num_iters):
        data = dataset.sample(mode='train')
        assert isinstance(data, list) or isinstance(
            data, tuple), "data must be instance or tuple"
        if not isinstance(data[0], torch.Tensor):
            data = [togpu(i) for i in data]

        output = agent.update(*data)
        train_outputs.append(output)

        if (batch_id + 1) % eval_episode == 0:
            agent.eval()
            optim = agent.optim # make sure that it's not optimized
            agent.optim = None

            train_info = merge_training_output(train_outputs)
            output['__stage'] = 'train'
            for fn in train_vis_fn:
                fn(train_info, train_outputs, data, output)

            train_outputs = []
            valid_outputs = []
            for batch_val_id in tqdm.trange(num_valid_iters):
                data = dataset.sample(mode='valid')
                if not isinstance(data[0], torch.Tensor):
                    data = [togpu(i) for i in data]
                output = agent.update(*data)
                valid_outputs.append(output)
            valid_info = merge_training_output(valid_outputs)
            output['__stage'] = 'valid'
            for fn in valid_vis_fn:
                fn(valid_info, valid_outputs, data, output)

            train_vis(train_info)
            valid_vis(valid_info)

            agent.train()
            agent.optim = optim
            if save_per_epsiode:
                agent.save(path, (batch_id+1)//eval_episode)
            else:
                agent.save(path)

    return agent
