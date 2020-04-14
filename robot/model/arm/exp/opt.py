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


def trainQACC(model, dataset_path):
    dataset = QACCDataset(dataset_path)

    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    l2 = nn.MSELoss()

    for epoch in range(100):
        for phase in ['train', 'valid']:
            num_iter = 100 if epoch == 'train' else 20
            outputs = []
            for i in tqdm.trange(num_iter):
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                data = dataset.sample(phase, 256)
                s = data[0][:, :4].double()
                qpos, qvel = s[:, :2], s[:, 2:4]
                a = data[1].double()
                qacc = data[2].double()

                # we assume that we don't train ee
                predict_qacc = model.qacc(qpos, qvel, a)
                loss = l2(predict_qacc, qacc)
                outputs.append(U.tocpu(loss))
                if phase == 'train':
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
            print(f'{phase} loss: ', np.mean(outputs))




def learnG():

    env = A.train_utils.make('acrobat2')
    model: ArmModel = build_diff_model(env, timestep=0.025, max_velocity=np.inf, damping=0.5)

    model.A.requires_grad = False
    model._M.requires_grad = False

    #dtype= model._G.dtype
    #dof = 2
    #G = [torch.rand((4,), dtype=dtype, device=model._G.device) for _ in range(dof)]  # G should be positive...
    #model._G.data = torch.stack(G)

    trainQACC(model, '/dataset/acrobat2')


if __name__ == "__main__":
    #main()
    learnG()

