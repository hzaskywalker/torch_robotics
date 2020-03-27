import torch
from torch import nn
import os
import cv2
import tqdm
import gym
import numpy as np


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
