import numpy as np
import torch
from robot.model.arm.exp.phys_model import ArmModel
from robot.model.arm.exp.arm_validator import get_env_agent, build_diff_model
from robot.model.arm.exp.learn_acrobat_qacc import trainQACC, QACCDataset

def learn_qacc(typeG='spatial'):
    optimize_A = False
    optimize_M = True # if optimize M will be better?
    optimize_G = True
    env, agent = get_env_agent(seed=None)
    model: ArmModel = build_diff_model(env, damping=3.0, typeG=typeG)
    dof = 7
    dataset = QACCDataset('/dataset/arm', small=False)

    print(model._G)
    def make_model(model):
        dtype= model._G.dtype
        if optimize_A:
            model._A.requires_grad = True
        else:
            model._A.data[:] = torch.tensor([1,0,0,0,0,0], dtype=dtype, device='cuda:0')
            model._A.requires_grad = False

        if optimize_M:
            model._M.requires_grad = True
            model._M.data[:] = torch.randn_like(model._M.data) * 0.1
        else:
            model._M.requires_grad = False

        if optimize_G:
            model._G.requires_grad = True
            if typeG == 'spatial':
                G = torch.stack([torch.rand((10,), dtype=dtype, device=model._G.device) for _ in range(dof)])
                G[...,3:6] = 0
                G[...,7:10] = 0
            else:
                G = torch.stack([torch.rand((4,), dtype=dtype, device=model._G.device) for _ in range(dof)])
            model._G.data = G
        else:
            model._G.requires_grad = False

        return model

    model = make_model(model)
    trainQACC(model, dataset, learn_ee=100., viewer=None, learn_qacc=1.,
                         optim_method='adam', epoch_num=40, loss_threshold=1e-10, lr=1e-3, num_train=1000, batch_size=256)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--diag', action='store_true')
    args = parser.parse_args()
    if args.diag:
        learn_qacc('diag')
    else:
        learn_qacc('spatial')
