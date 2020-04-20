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
    env = get_env_agent()[0]
    model: ArmModel = build_diff_model(env)
    dof = 7

    optimize_A = True
    optimize_M = True
    optimize_G = False

    dtype= model._G.dtype
    if optimize_A:
        model._A.requires_grad = True
        model._A.data[:] = torch.tensor([0., 0., 1., 0.0, 0, 0], dtype=dtype, device='cuda:0')
        #model._A.data += torch.randn_like(model._A.data) * 0.5
    else:
        model._A.requires_grad = False

    if optimize_M:
        model._M.requires_grad = True
        cc = model._M.data[0].clone()
        model._M.data[:] = torch.tensor(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
            dtype=dtype, device=model._M.device)
        model._M.data[0] = cc + torch.randn_like(cc) # xjb hack
    else:
        model._M.requires_grad = False

    if optimize_G:
        model._G.requires_grad = True
        G = [torch.rand((4,), dtype=dtype, device=model._G.device) for _ in range(dof)]  # G should be positive...
        model._G.data = torch.stack(G)
    else:
        model._G.requires_grad = False

    viewer = Viewer(model)
    trainQACC(model, '/dataset/arm', env, learn_ee=1., viewer=viewer, learn_qacc=0., optim_method='adam')
    #cemQACC_M(model, '/dataset/arm', param=model._A)

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    if not args.show:
        learnG()
    else:
        show()
