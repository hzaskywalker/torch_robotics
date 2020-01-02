import torch
from torch import nn
import os
import cv2
import tqdm
import gym
import numpy as np


def save(path, field, epochs):
    print('Saving.. ', path)
    #state = net.state_dict()
    if not os.path.isdir(path):
        os.mkdir(path)
    field = Field({i:field[i].state_dict() for i in field})
    field.epoch = epochs
    torch.save(field, os.path.join(path, 'ckpt.t7'))


def resume(path, dict):
    print('==> Resuming from checkpoint..: {}'.format(path))
    assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(path, 'ckpt.t7'))
    for i in dict:
        dict[i].load_state_dict(checkpoint[i])
    print('==> Loaded epoch: {}'.format(checkpoint['epoch']))
    return checkpoint['epoch']


def resume_if_exists(path, net):
    if path is not None and os.path.exists(os.path.join(path, 'ckpt.t7')):
        return resume(path, net)
    return 0


def test(envs, model, nstack, render=False, viewer=None):
    obs = envs.reset()
    if nstack > 1:
        obs = np.concatenate([obs*0] * (nstack-1)+[obs], axis=1)
    mb_dones = []
    mb_rewards = []
    nn = 0
    for i in tqdm.trange(2000):
        if render:
            obs = obs[None, :]
        actions = model.collector_step(np.float32(obs))[0]
        if render:
            actions = actions[0]
        l = obs
        obs, rewards, dones, _ = envs.step(actions)
        if render:
            dones = [dones]
        if render:
            if viewer is not None:
                import cv2
                img = viewer.render(obs)
                cv2.imshow('x', img)
                cv2.waitKey(0)
            else:
                envs.render()
        if nstack > 1:
            obs = np.concatenate(
                [l[:, :obs.shape[1] * (nstack-1)], obs], axis=1)
        mb_dones.append(dones)
        mb_rewards.append(rewards)

        if dones[0]:
            envs.reset()
            nn += 1
            if nn >= 8:
                break

        if dones[0] and i > 1600:
            break
    if not render:
        return np.sum(np.array(mb_rewards)[:, 0]) / np.sum(np.array(mb_dones)[:, 0])
    else:
        return np.sum(np.array(mb_rewards)) / np.sum(np.array(mb_dones))


def make_generator(batch_size, *args):
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


class Normalizer(nn.Module):
    def __init__(self, space, bound=1, show=True):
        nn.Module.__init__(self)
        low = np.maximum(-bound, space.low)
        high = np.minimum(bound, space.high)
        self.low = nn.Parameter(torch.tensor(
            low.reshape(-1)), requires_grad=False)
        self.high = nn.Parameter(torch.tensor(
            high.reshape(-1)), requires_grad=False)
        if show:
            print('low', self.low, space.low)
            print('high', self.high, space.high)
        self.oup_dim = np.prod(space.shape)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return (x[:, ] - self.low[None, :]) / (self.high - self.low)[None, :] - 0.5

    def denorm(self, x):
        x = x.view(x.size(0), -1)
        return (x + 0.5) * (self.high.detach() - self.low.detach())[None, :] + self.low.detach()[None, :]


class NormalizerV2(Normalizer):
    def __init__(self, x):
        nn.Module.__init__(self)
        x = x.view(x.size(0), -1)
        self.low = nn.Parameter(x.min(dim=0)[0], requires_grad=False)
        self.high = nn.Parameter(x.max(dim=0)[0], requires_grad=False)
        print('range', self.low, self.high)


def select_discrete_action(out, action, action_n):
    # for different action space
    assert len(action.shape) == 1
    out = out.view(out.size(0), action_n, -1)
    index = action[:, None, None].expand(out.size(0), 1, out.size(2)).long()
    return torch.gather(out, dim=1, index=index)[:, 0]


def batched_index_select(input, dim, index):
    assert len(index.shape) <= 2
    views = [input.shape[0]] + \
        [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)

def dict2field(d):
    out = Field(d)
    for key, val in out.items():
        if isinstance(val, dict):
            out[key] = dict2field(val)
    return out


def write_video(gen, path=None):
    out = None
    for img in gen:
        if path is not None:
            if out is None:
                out = cv2.VideoWriter(
                    path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (img.shape[1], img.shape[0]))
            out.write(img)
        else:
            cv2.imshow('x', img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    if out is not None:
        out.release()
