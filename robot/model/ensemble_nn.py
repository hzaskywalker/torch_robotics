# ensemble bayesian network
import torch
from torch import nn
from robot.utils import AgentBase
import numpy as np
import torch.nn.functional as F
from robot.utils.normalizer import Normalizer
from robot.utils import as_input


def swish(x):
    return x * torch.sigmoid(x)

from scipy.stats import truncnorm
truncnorm = truncnorm(-2, 2)

def truncated_normal(size, std):
    # import tensorflow as tf
    # We use TF to implement initialization function for neural network weight because:
    # 1. Pytorch doesn't support truncated normal
    # 2. This specific type of initialization is important for rapid progress early in training in cartpole

    # Do not allow tf to use gpu memory unnecessarily
    #cfg = tf.ConfigProto()
    #cfg.gpu_options.allow_growth = True

    #sess = tf.Session(config=cfg)
    #val = sess.run(tf.truncated_normal(shape=size, stddev=std))

    # Close the session and free resources
    #sess.close()
    trunc = truncnorm.rvs(size=size) * std

    return torch.tensor(trunc, dtype=torch.float32)


class ensemble_fc(nn.Module):
    def __init__(self, ensemble_size, in_features, out_features, swish=False):
        super(ensemble_fc, self).__init__()

        w = truncated_normal(size=(ensemble_size, in_features, out_features),
                             std=1.0 / (2.0 * np.sqrt(in_features)))

        self.w = nn.Parameter(w)
        self.b = nn.Parameter(torch.zeros(ensemble_size, 1, out_features, dtype=torch.float32))
        self.swish = swish

    def forward(self, inputs):
        # inputs (ensemble size, batch, in_feature)
        # w (ensemble size, in_feature, out_features)
        inputs = inputs.matmul(self.w) + self.b
        if self.swish:
            inputs = swish(inputs)
        return inputs


def ensemble_mlp(ensemble_size, in_features, out_features, num_layers, mid_channels):
    layers = []
    if num_layers == 1:
        layers.append(ensemble_fc(ensemble_size, in_features, out_features))
    else:
        layers.append(ensemble_fc(ensemble_size, in_features, mid_channels, swish=True))
        for i in range(num_layers-2):
            layers.append(ensemble_fc(ensemble_size, mid_channels, mid_channels, swish=True))
        layers.append(ensemble_fc(ensemble_size, mid_channels, out_features))
    return nn.Sequential(*layers)


class GaussianLayer(nn.Module):
    def __init__(self, out_features):
        super(GaussianLayer, self).__init__()

        self.out_features = out_features

        # TODO: need add some normalization term
        self.max_logvar = nn.Parameter(torch.ones(1, out_features // 2, dtype=torch.float32) / 2.0)
        self.min_logvar = nn.Parameter(- torch.ones(1, out_features // 2, dtype=torch.float32) * 10.0)

    def forward(self, inputs):
        mean = inputs[:, :, :self.out_features // 2]

        logvar = inputs[:, :, self.out_features // 2:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    def decay(self):
        return self.max_logvar.sum() - self.min_logvar.sum()



class EnBNN(nn.Module):
    # ensemble bayesian
    def __init__(self, ensemble_size, in_features, out_features, num_layers, mid_channels):
        super(EnBNN, self).__init__()
        self.ensemble_size = ensemble_size
        self.mlp = ensemble_mlp(ensemble_size, in_features, out_features * 2, num_layers, mid_channels)
        self.gaussian = GaussianLayer(out_features * 2)

    def forward(self, obs, action):
        # obs (ensemble, batch, dim_obs) or (batch, dim_obs)
        # action (ensemble, batch, action)
        inp = torch.cat((obs, action), dim=-1)
        if inp.shape == 2:
            inp = inp[None, :, :].expand(self.ensemble_size, -1, -1)
        return self.gaussian(self.mlp(inp))

    def var_reg(self):
        return self.gaussian.decay()

    def decay(self, weights=0.0001):
        if isinstance(weights, float):
            weights = [weights] * len(self.mlp)
        loss = 0
        for w, m in zip(weights, self.mlp):
            loss = w * (m.w ** 2).sum() / 2.0 + loss
        return loss


class EnBNNAgent(AgentBase):
    def __init__(self, lr, env, weight_decay=0.0002, var_reg=0.01,
                 ensemble_size=5, *args, **kwargs):
        state_prior = env.state_prior

        inp_dim = state_prior.inp_dim
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.forward_model = EnBNN(ensemble_size, inp_dim + action_dim, obs_dim, *args, **kwargs)

        self.obs_norm: Normalizer = Normalizer((inp_dim,))
        self.action_norm: Normalizer = Normalizer((action_dim,))

        super(EnBNNAgent, self).__init__(self.forward_model, lr)
        self.weight_decay = weight_decay
        self.var_reg = var_reg

        self.state_prior = env.state_prior # which is actually a config file of the environment
        self.batch_size = 32
        self.ensemble_size = ensemble_size

    def cuda(self):
        self.obs_norm.cuda()
        self.action_norm.cuda()
        return super(EnBNNAgent, self).cuda()

    def get_predict(self, s, a):
        inp = self.state_prior.encode(s)
        inp = self.obs_norm(inp)
        a = self.action_norm(a)
        mean, std = self.forward_model(inp, a)
        return self.state_prior.add(s, mean), std

    def rollout(self, s, a):
        # s (inp_dim)
        # a (pop, T, acts)
        if len(s.shape) == 1:
            s = s[None, :].expand(a.shape[0], -1)
        s = s[None, :].expand(self.ensemble_size, -1, -1)
        outs = []
        rewards = []
        for i in range(a.shape[1]):
            act = a[None, :, i].expand(self.ensemble_size, -1, -1)
            mean, std = self.get_predict(s, act)
            t = torch.rand_like(std) * std + mean # sample
            outs.append(t)
            rewards.append(self.state_prior.cost(s, act, t))
            s = t
        return torch.stack(outs, dim=2), torch.stack(rewards, dim=1).mean(dim=(0,1)) # get the reward amont time and B

    def update(self, s, a, t):
        if self.training:
            self.optim.zero_grad()
            self.obs_norm.fit(s)
            self.action_norm.fit(a)

        output = self.get_predict(s, a)
        loss = self.state_prior.dist(output, t)

        loss += self.var_reg * self.forward_model.var_reg()
        loss += self.forward_model.decay(self.weight_decay)

        loss = loss.mean(dim=(-1, -2))

        if self.training:
            loss.backward()
            self.optim.step()
        return {
            'loss': loss.detach().cpu().numpy()
        }
