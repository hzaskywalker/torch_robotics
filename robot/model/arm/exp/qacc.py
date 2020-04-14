import torch
import tqdm
import glob
import pickle
import os
import numpy as np
from robot import A, U, tr
from robot.model.arm.exp.sapien_validator import build_diff_model, ArmModel, get_env_agent, compute_qacc

def collect(file_name):
    import pickle
    with open(file_name, 'rb') as f:
        obs, actions, _ = pickle.load(f)
    env, agent = get_env_agent()
    assert env.dt < 1e-3
    qacc = []
    #for obs, action in zip(obs, actions):
    ran = range
    if file_name[-6:] == '/0.pkl':
        ran = tqdm.trange
    for i in ran(len(obs)):
        for o, a in zip(obs[i], actions[i]):
             qacc.append(compute_qacc(env, agent, o[:2], o[2:4], a.clip(-1, 1) * 50))
    return np.array(qacc).reshape(-1, actions.shape[1], 2), file_name

class QACCDataset:
    def __init__(self, dataset_path, valid_ratio=0.2):
        self.dataset_path = dataset_path
        self.path = os.path.join(dataset_path, 'qacc')
        if not os.path.exists(self.path):
            dataset = self.create()
            with open(self.path, 'wb') as f:
                pickle.dump(dataset, f)
        else:
            with open(self.path, 'rb') as f:
                dataset = pickle.load(f)
        self.qacc = dataset

        dd = A.train_utils.Dataset(dataset_path)
        self.device = dd.device
        self.obs, self.action = dd.obs, dd.action
        self.num_train = int(len(self.qacc) * (1-valid_ratio))
        self.qacc = self.qacc[dd.not_inf_mask][dd._rand_idx]
        print(self.obs.shape, self.action.shape, self.qacc.shape)

    def create(self):
        from multiprocessing import Pool
        p = Pool(20)
        qaccs = []
        for qacc, f in p.map(collect, [os.path.join(self.dataset_path, f'{i}.pkl') for i in range(20)]):
            print(f)
            qaccs.append(qacc)
        return np.concatenate(qaccs)

    def sample(self, mode='train', batch_size=256):
        if mode == 'train':
            idx = np.random.choice(self.num_train, batch_size)
        elif mode == 'valid':
            idx = np.random.choice(len(self.obs) - self.num_train, batch_size) + self.num_train
        else:
            raise NotImplementedError

        idx2 = np.random.randint(self.obs.shape[-2]-1, size=batch_size)
        #s = np.take_along_axis(self.obs[idx], idx2, axis=1)
        s = self.obs[idx, idx2]
        a = self.action[idx, idx2]
        ddq = self.qacc[idx, idx2]
        out = [torch.tensor(i, dtype=torch.float, device=self.device) for i in [s, a, ddq]]
        return out


def learn_qacc():
    env = A.train_utils.make('acrobat2')
    model: ArmModel = build_diff_model(env, timestep=0.025, max_velocity=np.inf, damping=0.5, dtype=torch.float64)

    def extract_state(obs):
        obs = obs['observation']
        return obs[:4]

    if False:
        obs = env.reset()
        for i in range(50):
            s = extract_state(obs)
            torque = env.action_space.sample()
            obs, _, _, _ = env.step(torque)
            t = extract_state(obs)
            ee = obs['observation'][-2:]

            print('torque', env.model.get_qf(), torque * 50,
                  env.model.compute_inverse_dynamics(env.model.get_qacc())+env.model.compute_passive_force())
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
        #torch.save(model, 'model_gt.pkl')
        #exit(0)
        A.exp.opt.train(model, dataset)

if __name__ == '__main__':
    learn_qacc()