# ensemble bayesian network
import torch
from torch import nn
from robot.utils import AgentBase
import numpy as np
import torch.nn.functional as F
from robot.utils.normalizer import Normalizer


# get the flat grads or params
def _get_flat_params_or_grads(network, mode='params'):
    """
    include two kinds: grads and params
    """
    attr = 'data' if mode == 'params' else 'grad'
    #return np.concatenate([getattr(param, attr).cpu().numpy().flatten() for param in network.parameters()])
    return torch.cat([getattr(param, attr).reshape(-1).cpu() for param in network.parameters()])


def _set_flat_params_or_grads(network, flat_params, mode='params'):
    """
    include two kinds: grads and params
    """
    attr = 'data' if mode == 'params' else 'grad'
    # the pointer
    pointer = 0
    for param in network.parameters():
        #getattr(param, attr).copy_(torch.tensor(flat_params[pointer:pointer + param.data.numel()]).view_as(param.data))
        getattr(param, attr).copy_(flat_params[pointer:pointer + param.data.numel()].view_as(param.data))
        pointer += param.data.numel()


def swish(x):
    return x * torch.sigmoid(x)

from scipy.stats import truncnorm
truncnorm = truncnorm(-2, 2)

def truncated_normal(size, std):
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

        self.max_logvar = nn.Parameter(torch.ones(1, out_features // 2, dtype=torch.float32) / 2.0)
        self.min_logvar = nn.Parameter(-torch.ones(1, out_features // 2, dtype=torch.float32) * 10.0)

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
            inp = inp[None, :, :]
        if inp.shape[0] != self.ensemble_size:
            inp = inp.expand(self.ensemble_size, -1, -1)
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
    def __init__(self, lr, encode_obs, add_state, compute_reward,
                 inp_dim, action_dim, oup_dim,
                 weight_decay=0.0002, var_reg=0.01, npart=20,
                 ensemble_size=5, pipe=None, master=False, *args, **kwargs):
        """
        encode_obs takes state as input and output the oup_dim
        state_add tasks state as input and oup_dim as input and output new state_dim
        action_dim is the action_dim
        """
        self.pipe = pipe
        self.master = master

        self.device = 'cpu'
        self.encode_obs = encode_obs
        self.add_state = add_state
        self.compute_reward = compute_reward

        self.forward_model = EnBNN(ensemble_size, inp_dim + action_dim, oup_dim, *args, **kwargs)

        self.npart = npart
        self.ensemble_size = ensemble_size
        assert self.npart % self.ensemble_size == 0 and self.npart > 0

        self.obs_norm: Normalizer = Normalizer((inp_dim,))
        self.action_norm: Normalizer = Normalizer((action_dim,))

        super(EnBNNAgent, self).__init__(self.forward_model, lr)
        self.weight_decay = weight_decay
        self.var_reg = var_reg

    def cuda(self):
        self.obs_norm.cuda()
        self.action_norm.cuda()
        self.device = 'cuda:0'
        return super(EnBNNAgent, self).cuda()

    def predict(self, s, a):
        inp = self.encode_obs(s)
        mean, log_var = self.forward_model(self.obs_norm(inp), self.action_norm(a))
        return self.add_state(s, mean), log_var

    def rollout(self, s, a, goal):
        # s (inp_dim)
        # a (pop, T, acts)
        with torch.no_grad():
            if len(s.shape) == 1:
                s = s[None, :].expand(a.shape[0], -1)
            s = s[None, :].expand(self.npart, -1, -1).reshape(self.ensemble_size, -1, *s.shape[1:])

            reward = 0
            for i in range(a.shape[1]):
                act = a[None, :, i].expand(self.npart, -1, -1).reshape(self.ensemble_size, -1, *a.shape[2:])
                mean, log_var = self.predict(s, act)
                t = torch.randn_like(log_var) * torch.exp(log_var * 0.5) + mean # sample
                reward = self.compute_reward(s, act, t, goal) + reward
                s = t

            return reward.reshape(self.ensemble_size, -1, a.shape[0]).mean(dim=(0, 1))

    def update_normalizer(self, batch, normalizer='obs'):
        if normalizer == 'batch':
            self.update_normalizer(batch[0], 'obs')
            self.update_normalizer(batch[0], 'action')
            return
        batch = torch.tensor(batch.reshape(-1, batch.shape[-1]), dtype=torch.float32, device=self.device)

        if normalizer == 'obs':
            batch = self.encode_obs(batch)

        s, sq, count = batch.sum(dim=0), (batch**2).sum(dim=0), len(batch)

        normalizer = self.obs_norm if normalizer=='obs' else self.action_norm
        if self.pipe is not None:
            self.pipe.send([s, sq, count])
            s, sq, count = self.pipe.recv()

        normalizer.add(s, sq, count)

    def update(self, s, a, t=None):
        if t is None:
            assert s.shape[1] == 2
            s, t = s[:, 0], s[:, 1]
            a = a[:, 0]

        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.float, device=self.device)
            a = torch.tensor(a, dtype=torch.float, device=self.device)
            t = torch.tensor(t, dtype=torch.float, device=self.device)

        if self.training:
            self.optim.zero_grad()

        mean, log_var = self.predict(s, a)

        inv_var = torch.exp(-log_var)
        loss = ((mean - t.detach()[None, :]) ** 2) * inv_var + log_var
        loss = loss.mean(dim=(-2, -1)).sum(dim=0) # sum across different models

        loss += self.var_reg * self.forward_model.var_reg()
        loss += self.forward_model.decay(self.weight_decay)

        if self.training:
            loss.backward()
            self.sync_grads(self.forward_model)
            self.optim.step()
        return {
            'loss': loss.mean().detach().cpu().numpy()
        }


    def set_params(self, params):
        assert isinstance(params, dict)
        _set_flat_params_or_grads(self.forward_model , params['model'], mode='params'),

    def get_params(self):
        return {
            'model': _get_flat_params_or_grads(self.forward_model, mode='params'),
        }

    def sync_grads(self, net):
        if self.pipe is not None:
            grad = _get_flat_params_or_grads(net, mode='grad')
            self.pipe.send(grad)
            grad = self.pipe.recv()
            _set_flat_params_or_grads(net, grad, mode='grad')
