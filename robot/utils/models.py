import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, relu=False, batch_norm=False, *args, **kwargs):
    models = [layer_init(nn.Conv2d(in_channels, out_channels,
                                   kernel_size, stride, padding=padding, *args, **kwargs))]
    if relu:
        models.append(nn.ReLU())
    if batch_norm:
        models.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*models)


def conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, relu=False, batch_norm=False, *args, **kwargs):
    models = [layer_init(nn.Conv1d(in_channels, out_channels,
                                   kernel_size, stride, padding=padding, *args, **kwargs))]
    if relu:
        models.append(nn.ReLU())
    if batch_norm:
        models.append(nn.BatchNorm1d(out_channels))
    return nn.Sequential(*models)


class fc(nn.Module):
    def __init__(self, in_channels, out_channels, relu=False, batch_norm=False):
        nn.Module.__init__(self)
        models = [nn.Linear(in_channels, out_channels)]
        if relu:
            models.append(nn.ReLU())
        if batch_norm:
            models.append(nn.BatchNorm1d(out_channels))
        self.models = nn.Sequential(*models)

    def forward(self, x):
        batch_size = None
        if len(x.shape) == 3:
            batch_size = x.size(0)
            x = x.view(x.size(0) * x.size(1), -1)
        x = self.models(x)
        if batch_size is not None:
            x = x.view(batch_size, -1, x.size(-1))
        return x


def deconv(in_channels, out_channels, kernel_size, stride=1, padding=0, relu=False, batch_norm=False, *args, **kwargs):
    models = [layer_init(nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size, stride, padding=padding, *args, **kwargs))]
    if relu:
        models.append(nn.ReLU())
    if batch_norm:
        models.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*models)

class Identity(nn.Module):
    def forward(self, x):
        return x

class BatchReshape(nn.Module):
    def __init__(self, *args):
        nn.Module.__init__(self)
        if isinstance(args[0], tuple):
            assert len(args) == 1
            self.shape = args[0]
        else:
            self.shape = args

    def forward(self, inp):
        return inp.view(inp.size(0), *self.shape)

class CNNEncoder84(nn.Module):
    def __init__(self, in_channels=4, feature_dim=512, batch_norm=False):
        super(CNNEncoder, self).__init__()
        self.feature_dim = feature_dim
        # self.feature_dim = 256
        self.conv1 = conv2d(in_channels, 32, kernel_size=8,
                            stride=4, relu=True, batch_norm=batch_norm)
        self.conv2 = conv2d(32, 64, kernel_size=4, stride=2,
                            relu=True, batch_norm=batch_norm)
        self.conv3 = conv2d(64, 64, kernel_size=3, stride=1,
                            relu=True, batch_norm=batch_norm)
        self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class CNNEncoder21(nn.Module):
    def __init__(self, in_channels=4, feature_dim=512, batch_norm=False):
        super(CNNEncoder21, self).__init__()
        if batch_norm:
            print("WARNNING: BATCH_NORM IS NOT SUPPORT FOR CURRENT ENCODER")
        self.feature_dim = feature_dim
        # self.feature_dim = 256
        self.conv1 = layer_init(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2)) # 10x10
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)) # 5x5
        self.conv3 = layer_init(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)) # 3x3
        self.fc4 = layer_init(nn.Linear(2 * 2 * 128, self.feature_dim))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class CNNEncoder64(nn.Module):
    def __init__(self, in_channels=4, feature_dim=512, batch_norm=False):
        nn.Module.__init__(self)
        self.main = nn.Sequential(
            conv2d(in_channels, 32, 4, 2, 1, relu=True, batch_norm=batch_norm), #32
            conv2d(32, 32, 4, 2, 1, relu=True, batch_norm=batch_norm), # 16
            conv2d(32, 64, 4, 2, 1, relu=True, batch_norm=batch_norm), # 8
            conv2d(64, 64, 4, 2, 1, relu=True, batch_norm=batch_norm), # 4
            conv2d(64, 256, 4, 2, 1, relu=True, batch_norm=batch_norm), # 2
            conv2d(256, 256, 4, 2, 1, relu=True, batch_norm=batch_norm), # 2
            conv2d(256, feature_dim, 1, relu=False, batch_norm=False)
        )

    def forward(self, x):
        output = self.main(x)
        return output.view(x.size(0), -1)

    @classmethod
    def test(cls):
        import numpy as np
        x = CNNEncoder64(1, 512).cuda()
        y = torch.Tensor(np.zeros(shape=(128, 1, 64, 64))).cuda()
        print(x(y).shape)


def CNNEncoder(inp_size, in_channels, feature_dim, batch_norm):
    if inp_size == 21:
        return CNNEncoder21(in_channels, feature_dim, batch_norm)
    elif inp_size == 64:
        return CNNEncoder64(in_channels, feature_dim, batch_norm)
    elif inp_size == 84:
        return CNNEncoder84(in_channels, feature_dim, batch_norm)
    else:
        raise NotImplementedError




class CNNDecoder21(nn.Module):
    def __init__(self, feature_dim, output_dim, batch_norm):
        nn.Module.__init__(self)
        self.linear = nn.Linear(feature_dim, 128)
        self.decoder = nn.Sequential(
            deconv(128, 128, kernel_size=3, stride=2,
                   relu=True, batch_norm=batch_norm),
            deconv(128, 64, kernel_size=5, stride=2,
                   relu=True, batch_norm=batch_norm),
            deconv(64, 32, kernel_size=5, stride=2,
                   relu=True, batch_norm=batch_norm),
            deconv(32, output_dim, kernel_size=5, stride=1, relu=True),
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.size(0), 128, 1, 1)
        return self.decoder(x)[:, :, 2:-2, 2:-2]  # very stupid here

    @classmethod
    def test(cls):
        print(cls)
        import tqdm
        import numpy as np
        model = CNNDecoder21(129, 5).cuda()

        for i in tqdm.trange(10000):
            x = torch.Tensor(np.random.random((128, 129))).cuda()
            ans = model(x)
        return


class CNNDecoder84(nn.Module):
    def __init__(self, feature_dim, output_dim, batch_norm):
        nn.Module.__init__(self)
        self.linear = nn.Linear(feature_dim, 5*5*128)
        self.decoder = nn.Sequential(
            deconv(128, 128, kernel_size=4, stride=1,
                   relu=True, batch_norm=batch_norm),
            deconv(128, 64, kernel_size=5, stride=2,
                   relu=True, batch_norm=batch_norm),
            deconv(64, 32, kernel_size=5, stride=2,
                   relu=True, batch_norm=batch_norm),
            deconv(32, output_dim, kernel_size=5, stride=2),
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.size(0), 128, 5, 5)
        return self.decoder(x)[:, :, :-1, :-1]  # very stupid here

    @classmethod
    def test(cls):
        print(cls)
        import tqdm
        import numpy as np
        model = CNNDecoder84(129, 5).cuda()

        for i in tqdm.trange(10000):
            x = torch.Tensor(np.random.random((128, 129))).cuda()
            ans = model(x)
        return


class CNNDecoder64(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm):
        nn.Module.__init__(self)

        self.decoder = nn.Sequential(
            BatchReshape(in_channels, 1, 1),
            deconv(in_channels, 128, 1, relu=True, batch_norm=batch_norm),
            deconv(128, 64, 4, relu=True, batch_norm=batch_norm),
            deconv(64, 64, 4, 2, 1, relu=True, batch_norm=batch_norm),
            deconv(64, 32, 4, 2, 1, relu=True, batch_norm=batch_norm),
            deconv(32, 32, 4, 2, 1, relu=True, batch_norm=batch_norm),
            deconv(32, out_channels, 4, 2, 1),
        )
        self.out_channels = out_channels

    def forward(self, inp):
        return self.decoder(inp)


class Constant(nn.Module):
    def __init__(self, tensor):
        nn.Module.__init__(self)
        self.out = nn.Parameter(tensor)

    def forward(self, *input, **kwargs):
        return self.out


def CNNDecoder(oup_size, feature_dim, output_dim, batch_norm):
    if oup_size == 21:
        return CNNDecoder21(feature_dim, output_dim, batch_norm=batch_norm)
    elif oup_size == 84:
        return CNNDecoder84(feature_dim, output_dim, batch_norm=batch_norm)
    elif oup_size == 64:
        return CNNDecoder64(feature_dim, output_dim, batch_norm=batch_norm)
    else:
        raise NotImplementedError