import torch
import tqdm
import glob
import pickle
import os
import numpy as np
from robot import A, U, tr
from robot.model.arm.exp.sapien_validator import compute_qacc

def collect(file_name):
    import pickle
    with open(file_name, 'rb') as f:
        obs, actions, _ = pickle.load(f)
    dof = actions.shape[-1]
    if 'arm' in file_name:
        from robot.model.arm.exp.arm_validator import get_env_agent
        env, agent = get_env_agent()
    else:
        from robot.model.arm.exp.sapien_validator import get_env_agent
        env, agent = get_env_agent()
    assert env.dt < 1e-3
    qacc = []
    #for obs, action in zip(obs, actions):
    ran = range
    if file_name[-6:] == '/0.pkl':
        ran = tqdm.trange
    for i in ran(len(obs)):
        for o, a in zip(obs[i], actions[i]):
            qacc.append(compute_qacc(env, agent, o[:dof], o[dof:dof*2], a.clip(-1, 1) * 50))
    return np.array(qacc).reshape(-1, actions.shape[1], dof), file_name

class QACCDataset:
    def __init__(self, dataset_path, valid_ratio=0.2, small=False):
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
        if small == True:
            self.num_train = 30
        print("TRAINING SIZE", self.obs[:self.num_train].shape)
        self.qacc = self.qacc[dd.not_inf_mask][dd._rand_idx]
        print(self.obs.shape, self.action.shape, self.qacc.shape)

    def permute(self):
        idx = np.arange(len(self.obs))
        self._rand_idx = np.random.permutation(idx)
        self.obs = self.obs[self._rand_idx]
        self.action = self.action[self._rand_idx]
        self.qacc = self.qacc[self._rand_idx]

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
