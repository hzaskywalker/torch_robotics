import numpy as np
import torch
from robot.model.arm.exp.phys_model import ArmModel
from robot.model.arm.exp.arm_validator import get_env_agent, build_diff_model
from robot.model.arm.exp.sapien_validator import compute_qacc
from robot.model.arm.exp.learn_acrobat_qacc import trainQACC, QACCDataset, ee_loss, Viewer as AcrobatViewer, cemQACC_G

def learn_qacc():
    optimize_A = False
    optimize_M = True
    optimize_G = True
    env, agent = get_env_agent(seed=None)
    model: ArmModel = build_diff_model(env, damping=3.0)
    dof = 7
    dataset = QACCDataset('/dataset/arm', small=False)

    #from robot import A
    #env = A.exp.sapien_validator.get_env_agent()[0]
    #model: ArmModel = A.exp.sapien_validator.build_diff_model(env, timestep=0.025, max_velocity=np.inf, damping=0.5)
    #dof = 2
    #dataset = QACCDataset('/dataset/acrobat2', small=False)
    print(model._G)

    def make_model(model):
        dtype= model._G.dtype
        if optimize_A:
            model._A.requires_grad = True
            #model._A.data[:] = torch.randn_like(model._A.data) * 0.1
        else:
            model._A.requires_grad = False

        if optimize_M:
            model._M.requires_grad = True
            #model._M.data[:] = torch.randn_like(model._M.data) * 0.1
        else:
            model._M.requires_grad = False

        if optimize_G:
            model._G.requires_grad = True
            GG = model._G.data.clone()
            G = [torch.rand((4,), dtype=dtype, device=model._G.device) for _ in range(dof)]
            model._G.data = torch.stack(G)
            #model._G.data = GG
        else:
            model._G.requires_grad = False

        return model

    model = make_model(model)
    trainQACC(model, dataset, learn_ee=1., viewer=None, learn_qacc=1.,
                         optim_method='adam', epoch_num=40, loss_threshold=1e-10, lr=1e-3, num_train=1000, batch_size=256)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    learn_qacc()
