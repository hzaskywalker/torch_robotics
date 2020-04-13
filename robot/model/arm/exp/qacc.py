import torch
import numpy as np
from robot.model.arm.exp.phys_model import ArmModel
from robot import A, U, tr

def build_diff_model(env, timestep=0.01):
    model = ArmModel(2, max_velocity=100, timestep=timestep).cuda()

    model._M.data = torch.tensor(
        np.array(
            [[[1, 0, 0, 0],[0, 1, 0, -0.25], [0, 0, 1, 0], [0, 0, 0, 1]],
             [[1, 0, 0, 0], [0, 1, 0, -0.5], [0, 0, 1, 0], [0, 0, 0, 1]],
             [[1, 0, 0, 0], [0, 1, 0, -0.25], [0, 0, 1, 0], [0, 0, 0, 1]],]
        ), dtype=torch.float64, device='cuda:0')

    w, q = [0, 0, 1], [0, 0.25, 0]
    screw1 = w + (-np.cross(w, q)).tolist()
    model.A.data = torch.tensor([screw1, screw1], dtype=torch.float64, device='cuda:0')
    for idx, i in enumerate(env.unwrapped.model.get_links()[1:]):
        model._G.data[idx] = torch.tensor([i.inertia[0], i.inertia[2], i.inertia[1], i.get_mass()], dtype=torch.float64, device='cuda:0')
    return model

def learn_qacc():
    env = A.train_utils.make('acrobat2')
    model = build_diff_model(env)

    def extract_state(obs):
        obs = obs['observation']
        return obs[:4]

    if True:
        obs = env.reset()
        for i in range(50):
            s = extract_state(obs)
            torque = env.action_space.sample()
            obs, _, _, _ = env.step(torque)
            t = extract_state(obs)
            ee = obs['observation'][-2:]

            print('torque', env.model.get_qf(), torque * 50,
                  env.model.compute_inverse_dynamics(env.model.get_qacc()))
            print('torque+force', env.model.get_qf() - env.model.compute_passive_force(), env.model.compute_passive_force())
            q = U.togpu(t[:2], torch.float64)[None,:]
            dq = U.togpu(t[2:4], torch.float64)[None,:]
            ddq = U.togpu(env.model.get_qacc(), torch.float64)[None,:]
            predict_qf = tr.inverse_dynamics(q, dq, ddq, *model.get_parameters(q))
            print('predict torque', U.tocpu(predict_qf[0]))

            predict_qacc = U.tocpu(model.qacc(q, dq, U.togpu(torque, torch.float64)[None,:])[0])

            print('qacc', predict_qacc, env.model.get_qacc())
            print()

    else:
        dataset = A.train_utils.Dataset('/dataset/acrobat2')
        A.exp.phys_model.train(model, dataset)

if __name__ == '__main__':
    learn_qacc()