import torch
import numpy as np
from robot.model.arm.exp.phys_model import ArmModel
from robot.model.arm.exp.arm_validator import get_env_agent, build_diff_model
from robot import U
from robot.model.arm.exp.learn_acrobat_qacc import trainQACC, QACCDataset, ee_loss, Viewer as AcrobatViewer

def cemQACC_M(model, dataset_path, env=None, torque_norm=50, param=None):
    # try to understand the optimization difficulties
    dataset = QACCDataset(dataset_path)

    mean = U.tocpu(param.data)
    std = mean * 0 + 1.
    #mean[...,3]=1

    for i in range(40):
        population = np.random.normal(size=(1000, *mean.shape)) *std[None, :] + mean[None, :] # batch_size = 500
        values = []
        data = dataset.sample('train', 1024)
        for i in population:
            param.data = torch.tensor(i, dtype=torch.float64, device=model._G.device)
            values.append(U.tocpu(ee_loss(data, model)))

        values = np.array(values)
        elite_id = values.argsort()[:20]
        elites = population[elite_id]
        _mean, _std = elites.mean(axis=0), elites.std(axis=0)
        #mean = mean * 0.5 + _mean * 0.5
        #std = std * 0.5 + _std * 0.5
        mean, std = _mean, _std
        print(mean, values[elite_id].mean())

class Viewer(AcrobatViewer):
    def __init__(self, model):
        super(Viewer, self).__init__(model, scale=0.1)
        for i in range(6):
            self.screw.add_shapes(None)

    def set_camera(self, r):
        r.add_point_light([2, 2, 2], [255, 255, 255])
        r.add_point_light([2, -2, 2], [255, 255, 255])
        r.add_point_light([-2, 0, 2], [255, 255, 255])

        r.set_camera_position(2.2, -0.5, 1.2)
        r.set_camera_rotation(-3.14 - 0.5, -0.2)

    def render(self, mode):
        self.screw.update(U.tocpu(self.model.M), U.tocpu(self.model.A))
        for i in self.screw.loop(5):
            self.r.render(mode)

def learnG():
    env = get_env_agent(seed=None)[0]
    dof = 7

    optimize_A = True
    optimize_M = True
    optimize_G = False
    model: ArmModel = build_diff_model(env)

    def make_model(model, init_A=np.array([0., 1., 0., 0.0, 0, 0]) ):
        dtype= model._G.dtype
        if optimize_A:
            model._A.requires_grad = True
            model._A.data[:] = torch.tensor(init_A, dtype=dtype, device='cuda:0')
            #model._A.data[:] = torch.randn_like(model._A.data) * 0.1
        else:
            model._A.requires_grad = False

        if optimize_M:
            model._M.requires_grad = True
            #model._M.data[:] = torch.tensor(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
            #    dtype=dtype, device=model._M.device)
            model._M.data[:] = torch.randn_like(model._M.data) * 0.1
            #model._M.data[0,3,3] = cc[3,3]#torch.randn_like(cc) # xjb hack, in fact, better initialization may help
        else:
            model._M.requires_grad = False

        if optimize_G:
            model._G.requires_grad = True
            G = [torch.rand((4,), dtype=dtype, device=model._G.device) for _ in range(dof)]  # G should be positive...
            model._G.data = torch.stack(G)
        else:
            model._G.requires_grad = False

        return model

    dataset = QACCDataset('/dataset/arm', small=False)
    viewer = Viewer(model)
    #viewer = None
    losses = []
    #for i in range(30):
    for i in range(10):
        #initA = [[*np.random.normal(size=(3,)), 0, 0, 0] for j in range(7)]
        initA = [*np.random.normal(size=(3,)), 0, 0, 0]
        #initA = [2.7162355469095885, 1.5762245473496705, 0.34537523430899664, 0, 0, 0]
        model = make_model(model, initA)
        dataset.permute()
        losses += [trainQACC(model, dataset, learn_ee=1., viewer=viewer, learn_qacc=0.,
                        optim_method='adam', epoch_num=5, loss_threshold=5e-5, lr=0.01)]
        print(losses[-1])
        print('ACC', np.mean(np.array(losses)< 5e-5) )
    #cemQACC_M(model, '/dataset/arm', param=model._A)
    print(losses)



from torch import nn
from robot.utils.models import MLP
class ParamNet(nn.Module):
    def __init__(self, inp_dim, oup_dim, z_dim=10, mid_channels=32, num_layers=2, beta=1.):
        super(ParamNet, self).__init__()
        self.inp_dim = inp_dim
        self.encoder = MLP(inp_dim, z_dim*2, mid_channels, num_layers=num_layers)
        self.decoder = MLP(z_dim, oup_dim, mid_channels, num_layers=num_layers)
        self.beta = beta
        self.z_dim = z_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_divergence(self, mu, logvar):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD

    def forward(self, inp):
        tmp = self.encoder(inp.float())
        mu, logvar = tmp[..., :self.z_dim], tmp[..., self.z_dim:]
        z = self.reparameterize(mu, logvar)
        output = self.decoder(z)
        return output.double(), mu, logvar, self.kl_divergence(mu, logvar).double()
    
class ParamModel(nn.Module):
    def __init__(self, params, phys):
        super(ParamModel, self).__init__()
        self.params = params
        self.phys = phys
        self._A_shape = (7, 6)
        self._M_shape = (8, 4, 4)
        self._A_dim = np.prod(self._A_shape)

    def parameters(self, recurse: bool = ...):
        from itertools import chain
        return chain(self.params.parameters(), (self.phys._G,))

    def fk(self, inp):
        # Let's see what would happen
        inp2 = torch.cat((torch.cos(inp), torch.sin(inp)), dim=-1)
        AM, mu, logvar, kl_dist = self.params(inp2)
        del self.phys._A
        del self.phys._M

        self.phys._A = AM[..., :self._A_dim].reshape(-1, *self._A_shape) * 0.1
        self.phys._M = AM[..., self._A_dim:].reshape(-1, *self._M_shape) * 0.1
        return self.phys.fk(inp), kl_dist


def VAETraining(model, dataset, viewer=None, torque_norm=50,
              learn_ee=0., learn_qacc=1., epoch_num=1000, loss_threshold=1e-10, lr=0.001, beta=1.):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    import tqdm
    from robot.model.arm.exp.learn_acrobat_qacc import l2
    assert learn_qacc < 1e-10

    for epoch in range(epoch_num):
        for phase in ['train', 'valid']:
            num_iter = 1000 if phase == 'train' else 200
            outputs = []
            kl_losses = []
            for _ in tqdm.trange(num_iter):
                if phase == 'train':
                    model.train()
                    optim.zero_grad()
                else:
                    model.eval()

                data = dataset.sample(phase, 256)

                dof = data[1].shape[-1]
                s = data[0][:, :dof].double()
                ee_label = data[0][:, -3:].double()
                try:
                    predict_ee, kl_loss = model.fk(s)
                except AssertionError:
                    continue
                assert predict_ee.shape == ee_label.shape
                ee_loss = l2(predict_ee, ee_label)

                beta += 1e-2 - float(ee_loss)
                beta = max(beta, 0)

                loss = ee_loss * learn_ee + kl_loss * beta
                if phase == 'train':
                    loss.backward()
                    optim.step()

                outputs.append(U.tocpu(ee_loss))
                kl_losses.append(U.tocpu(kl_loss))

            mean_loss = np.mean(outputs)
            print(f'{phase} loss: ', mean_loss)
            print(f'{phase} kl loss: ', np.mean(kl_losses))

            if phase == 'valid':
                if viewer is not None:
                    viewer.render(mode='human')
                    viewer.r.save('tmp.pkl')
                if mean_loss < loss_threshold:
                    return mean_loss

    return mean_loss


def learn_with_VAE():
    env = get_env_agent(seed=None)[0]
    dof = 7

    phys: ArmModel = build_diff_model(env)
    oup_dim = np.prod(phys._A.shape) + np.prod(phys._M.shape)

    dataset = QACCDataset('/dataset/arm', small=False)
    class Viewer2(Viewer):
        def render(self, mode):
            model.fk(U.togpu([0]*dof)[None,:])
            self.screw.update(U.tocpu(model.phys.M[0]), U.tocpu(model.phys.A[0]))
            for i in self.screw.loop(5):
                self.r.render(mode)
    viewer = Viewer2(phys)

    losses = []
    for i in range(100):
        param = ParamNet(dof * 2, oup_dim, 32, 256, num_layers=4).cuda()
        model = ParamModel(param, phys)
        dataset.permute()
        loss =  VAETraining(model, dataset, learn_ee=1., viewer=viewer, learn_qacc=0.,
                          epoch_num=10, loss_threshold=5e-5, lr=0.01, beta=1)
        losses.append(loss)
        print('ACC', (np.array(losses)<5e-5).mean())
    print(losses)

class MLP2(nn.Module):
    def __init__(self, dof):
        nn.Module.__init__(self)
        self.model = MLP(dof*2, 3, 400, 5)

    def fk(self, inp):
        inp = torch.cat((torch.cos(inp), torch.sin(inp)), dim=-1).float()
        return self.model(inp).double()

def learn_MLP():
    dataset = QACCDataset('/dataset/arm', small=True)
    model = MLP2(7).cuda()
    trainQACC(model, dataset, learn_ee=1., viewer=None, learn_qacc=0.,
              optim_method='adam', epoch_num=5, loss_threshold=1e-10, lr=0.01)


def show():
    from robot.renderer import Renderer
    r = Renderer.load('tmp.pkl')
    import copy
    r2 = copy.deepcopy(r)

    env = get_env_agent()[0]
    model: ArmModel = build_diff_model(env)

    screw = r.get('screw')
    A, B = U.tocpu(screw.M[0]), U.tocpu(screw.A[0])
    C, D = U.tocpu(model.M), U.tocpu(model.A)

    screw.update(A, B)

    r.set_camera_position(2.2, -0.5, 1.2)
    r2.set_camera_position(2.2, -0.5, 1.2)

    for q in screw.loop(K=100):
        img1 = r.render(mode='rgb_array')
        screw.update(C, D)
        screw.set_pose(q)
        img2 = r.render(mode='rgb_array')
        screw.update(A, B)
        import cv2
        cv2.imshow('x', np.concatenate((img1, img2), axis=1))
        cv2.waitKey(1)


if __name__ == '__main__':
    # 6/30 failing

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--vae', action='store_true')
    parser.add_argument('--mlp', action='store_true')
    args = parser.parse_args()
    if args.mlp:
        learn_MLP()
        exit(0)
    if not args.show:
        if not args.vae:
            learnG()
        else:
            learn_with_VAE()
    else:
        show()
