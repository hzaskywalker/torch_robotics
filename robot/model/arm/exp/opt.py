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
            print(data[0][0, 0, -2:])
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

def qacc_loss(data, model, torque_norm=50):
    s = data[0][:, :4].double()
    qpos, qvel = s[:, :2], s[:, 2:4]
    a = data[1].double()
    qacc = data[2].double()

    predict_qacc = model.qacc(qpos, qvel, a)

    loss = l2(predict_qacc / torque_norm, qacc / torque_norm)
    return loss


def cemQACC_G(model, dataset_path, env=None, torque_norm=50):
    # try to understand the optimization difficulties
    dataset = QACCDataset(dataset_path)

    mean = U.tocpu(model._G.data)
    std = mean * 0 + 1.
    #mean[...,3]=1

    for i in range(40):
        population = np.random.normal(size=(500, *mean.shape)) *std[None, :] + mean[None, :] # batch_size = 500
        values = []
        for i in population:
            data = dataset.sample('train', 1024)
            model._G.data = torch.tensor(i, dtype=torch.float64, device=model._G.device)

            values.append(U.tocpu(qacc_loss(data, model, torque_norm)))

        values = np.array(values)
        elite_id = values.argsort()[:10]
        elites = population[elite_id]
        _mean, _std = elites.mean(axis=0), elites.std(axis=0)
        #mean = mean * 0.5 + _mean * 0.5
        #std = std * 0.5 + _std * 0.5
        mean, std = _mean, _std
        print(mean, values[elite_id].mean())


def trainQACC(model, dataset_path, env=None, torque_norm=50):
    dataset = QACCDataset(dataset_path)

    optim = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1000):
        for phase in ['train', 'valid']:
            num_iter = 1000 if phase == 'train' else 200
            outputs = []
            for i in tqdm.trange(num_iter):
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                data = dataset.sample(phase, 1024)
                loss = qacc_loss(data, model, torque_norm)
                outputs.append(U.tocpu(loss))
                if phase == 'train':
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

            if phase == 'valid':
                print("learned G:", model.G)
            else:
                print('grad', model._G.grad)
            print(f'{phase} loss: ', np.mean(outputs))


def learnG():

    #env = A.train_utils.make('acrobat2')
    env = A.exp.sapien_validator.get_env_agent()[0]
    model: ArmModel = build_diff_model(env, timestep=0.025, max_velocity=np.inf, damping=0.5)
    #model, env = None, None

    model.A.requires_grad = False
    model._M.requires_grad = False

    dtype= model._G.dtype
    dof = 2

    G = [torch.rand((4,), dtype=dtype, device=model._G.device) for _ in range(dof)]  # G should be positive...
    model._G.data = torch.stack(G)

    #trainQACC(model, '/dataset/acrobat2', env)
    cemQACC_G(model, '/dataset/acrobat2')


if __name__ == "__main__":
    #main()
    learnG()

