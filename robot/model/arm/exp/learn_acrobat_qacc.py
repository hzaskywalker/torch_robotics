import torch
import tqdm
import numpy as np
from torch import nn
from robot import U, A
from robot.model.arm.exp.sapien_validator import ArmModel, build_diff_model
from robot.model.arm.exp.qacc import QACCDataset

# this file is going to find the suitable prior and the optimization methods for the robot arm,
# at least for the two link robot arm

# first, let's see if it's possible to find the G matrices by gradient descent.

# second, let's see if it's possible to optimize G and the coordinate matrix at the same time for one-link

class AngleLoss(nn.Module):
    def forward(self, predict, label):
        diff = torch.abs(predict - label)
        return (torch.min(diff, 2 * np.pi - diff) ** 2).mean()


def train(model, dataset):
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    loss_fn2 = AngleLoss()

    def eval_predict(output, t):
        predict, ee = output
        dq_loss = loss_fn(predict[..., 2:], t[..., 2:4])/1000
        q_loss = loss_fn2(predict[..., :2], t[..., :2])
        ee_loss = loss_fn(ee[..., :2], t[..., -2:])
        return dq_loss, q_loss, ee_loss

    def validate(num_iter=20):
        model.eval()
        total = 0
        for i in tqdm.trange(num_iter):
            data = dataset.sample('valid', 256, timestep=2)
            s = data[0][:, 0, :4].double()
            t = data[0][:, 1].double()

            a = data[1][:, 0].double()
            dq,q,ee= eval_predict(model(s[:, :], a), t)
            total += dq+q+ee  # action_range = 50
        model.train()
        return U.tocpu(total/num_iter)

    import tqdm
    averages = []
    for i in tqdm.trange(10000):
        data = dataset.sample('train', 256, timestep=2)
        s = data[0][:, 0, :4].double()
        t = data[0][:, 1].double()
        a = data[1][:, 0].double()

        optim.zero_grad()
        predict, ee = model(s, a)
        losses = eval_predict((predict, ee), t)
        averages.append([U.tocpu(j) for j in losses])
        loss = sum(losses) # action_range = 50
        loss.backward()
        optim.step()

        if i % 100 == 0:
            print("learned G:", model.G)
            print("learned M:", model.M)
            print("learned A:", model.A)
            #print("env G:", mm.G[1])
            print("dq: {}, q: {}, ee: {}".format(*np.mean(averages, axis=0)))
            print('mse loss', U.tocpu(loss), predict[0], t[0])
            print('valid mse loss', validate(5))

            averages = []


l2 = nn.MSELoss()
def relative_l2(a, b):
    return (((a-b)/(b + (b.abs() < 1e-2).float())).abs()).mean()

def qacc_loss(data, model, torque_norm=50):
    dof = data[1].shape[-1]
    s = data[0][:, :dof * 2].double()
    qpos, qvel = s[:, :dof], s[:, dof:dof+dof]
    a = data[1].double()
    qacc = data[2].double()

    predict_qacc = model.qacc(qpos, qvel, a)

    loss = relative_l2(predict_qacc / torque_norm, qacc / torque_norm)
    return loss

def ee_loss(data, model):
    dof = data[1].shape[-1]
    s = data[0][:, :dof].double()
    predict_ee = model.fk(s)
    if dof == 2:
        loss = l2(predict_ee[:, :2], data[0][:, -2:].double()) * 2 + (predict_ee[..., 2] ** 2).mean()
    else:
        loss = l2(predict_ee, data[0][:, -3:].double())
    return loss


def cemQACC_G(model, dataset_path, env=None, torque_norm=50):
    # try to understand the optimization difficulties
    dataset = QACCDataset(dataset_path)

    mean = U.tocpu(model._G.data)
    std = mean * 0 + 1.
    #mean[...,3]=1

    for i in range(40):
        population = np.random.normal(size=(1000, *mean.shape)) *std[None, :] + mean[None, :] # batch_size = 500
        values = []
        data = dataset.sample('train', 1024)
        for i in population:
            model._G.data = torch.tensor(i, dtype=torch.float64, device=model._G.device)
            values.append(U.tocpu(qacc_loss(data, model, torque_norm)))

        values = np.array(values)
        elite_id = values.argsort()[:20]
        elites = population[elite_id]
        _mean, _std = elites.mean(axis=0), elites.std(axis=0)
        #mean = mean * 0.5 + _mean * 0.5
        #std = std * 0.5 + _std * 0.5
        mean, std = _mean, _std
        print(mean, values[elite_id].mean())


class Viewer:
    def __init__(self, model, scale=0.3):
        from robot.renderer import Renderer
        r = Renderer()
        self.set_camera(r)

        self.model = model
        self.screw = r.screw_arm(U.tocpu(model.M), U.tocpu(model.A), scale=scale, name='screw')
        self.r = r

    def set_camera(self, r):
        r.line([-3, 0, 0], [3, 0, 0], 0.02, (255, 255, 255))
        r.add_point_light([0, 2, 2], [255, 255, 255], intensity=100)
        r.add_point_light([0, -2, 2], [255, 255, 255], intensity=100)

        r.set_camera_position(0, 0, 3)
        r.set_camera_rotation(1.57, -1.57)
        r.axis(r.identity())

    def render(self, mode):
        self.screw.update(U.tocpu(self.model.M), U.tocpu(self.model.A))
        for i in self.screw.loop():
            self.r.render(mode)



def trainQACC(model, dataset_path, env=None, viewer=None, torque_norm=50, learn_ee=0., learn_qacc=1., optim_method='adam'):
    dataset = QACCDataset(dataset_path)
    if optim_method == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=0.001)
    else:
        optim = torch.optim.LBFGS(model.parameters(), lr=0.01)

    for epoch in range(1000):
        for phase in ['train', 'valid']:
            num_iter = 1000 if phase == 'train' else 200
            outputs = []
            ee_losses = []
            for i in tqdm.trange(num_iter):
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                def closure():
                    if phase == 'train':
                        optim.zero_grad()
                    data = dataset.sample(phase, 256)

                    if learn_qacc > 0:
                        loss = qacc_loss(data, model, torque_norm) * learn_qacc
                    else:
                        loss = model._A.mean() * 0 + model._M.mean() * 0+ model._G.mean() * 0

                    if learn_ee > 0:
                        ee = ee_loss(data, model)
                        loss = loss + ee * learn_ee
                        ee_losses.append(U.tocpu(ee))

                    if phase == 'train':
                        loss.backward()
                    return loss

                if optim_method == 'adam':
                    loss = closure()
                    if phase == 'train':
                        optim.step()
                else:
                    optim.zero_grad()
                    loss = optim.step(closure)
                    print(loss)
                outputs.append(U.tocpu(loss))

            if phase == 'valid':
                print("learned G:", model.G)
                print("learned M:", model.M)
                print("learned A:", model.A)
                viewer.render(mode='human')
                viewer.r.save('tmp.pkl')
            else:
                print('grad G', model._G.grad)
                print('grad M', model._M.grad)
                print('grad A', model.A.grad)
            print(f'{phase} loss: ', np.mean(outputs))
            if len(ee_losses) > 0:
                print(f'{phase} ee loss: ', np.mean(ee_losses))


def learnG():
    env = A.exp.sapien_validator.get_env_agent()[0]
    model: ArmModel = build_diff_model(env, timestep=0.025, max_velocity=np.inf, damping=0.5)
    dof = 2

    optimize_A = True
    optimize_M = True
    optimize_G = True

    dtype= model._G.dtype
    if optimize_A:
        model._A.requires_grad = True
        model._A.data[:] = torch.tensor([0.5, 0.5, 0.5, 0.0, 0, 0], dtype=dtype, device='cuda:0')
    else:
        model._A.requires_grad = False

    if optimize_M:
        model._M.requires_grad = True
        model._M.data = torch.tensor(np.array([
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] for _ in range(dof+1)]),
            dtype=dtype, device=model._M.device)
    else:
        model._M.requires_grad = False

    if optimize_G:
        model._G.requires_grad = True
        G = [torch.rand((4,), dtype=dtype, device=model._G.device) for _ in range(dof)]  # G should be positive...
        model._G.data = torch.stack(G)
    else:
        model._G.requires_grad = False

    viewer = Viewer(model)
    trainQACC(model, '/dataset/acrobat2', env, learn_ee=1., viewer=viewer)
    #cemQACC_G(model, '/dataset/acrobat2')


def show():
    from robot.renderer import Renderer
    r = Renderer.load('tmp.pkl')

    screw = r.get('screw')
    if False:
        env = A.exp.sapien_validator.get_env_agent()[0]
        model: ArmModel = build_diff_model(env, timestep=0.025, max_velocity=np.inf, damping=0.5)

        screw.update(U.tocpu(model.M), U.tocpu(model.A))

    screw.update(U.tocpu(screw.M[0]), U.tocpu(screw.A[0]))
    print(screw.M)
    print(screw.A)

    for _ in screw.loop(K=100):
        r.render(mode='human')


if __name__ == "__main__":
    learnG()
    #show()
