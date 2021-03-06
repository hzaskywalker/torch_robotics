# robot arm model
# we will use envs.hyrule.rl_env.ArmReachGoal to generate enough trianing data...
import robot
import os
import glob
import pickle
import tqdm
import torch
import numpy as np


# easier for parallel
def data_collector(env, make_policy, num_episode, timestep, path, make=None, use_tqdm=False, seed=None):
    if isinstance(env, str):
        env = make(env)

    if seed is not None:
        np.random.seed(seed)

    policy = make_policy(env)
    ran = tqdm.trange if use_tqdm else range

    observations = actions = None

    for i in ran(num_episode):
        obs = env.reset()
        for j in range(timestep):
            action = policy(obs)
            if isinstance(obs, dict):
                obs = obs['observation']

            if observations is None:
                observations = np.empty([num_episode, timestep + 1, obs.shape[-1]])
                actions = np.empty([num_episode, timestep, action.shape[-1]])

            observations[i, j] = obs
            actions[i, j] = action

            obs, reward, done, _ = env.step(action)
            observations[i, j+1] = obs['observation'] #TODO: hack here
            if done:
                break

    with open(path, 'wb') as f:
        print('saving... ', path)
        print('obs max', observations.max(axis=(0, 1)))
        pickle.dump([observations, actions, None], f)


def batch_collector(env, policy, path, num_episode=25):
    p = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    for idx, p in enumerate(p):
        obss, acts = [], []
        for _ in tqdm.trange(num_episode):
            obs = env.reset()
            observations = [obs['observation']]
            actions = []
            for j in tqdm.trange(50):
                action = policy(obs).clip(-1, 1)
                rand_action = np.random.random(action.shape) * 2 - 1 #(-1, 1)
                mask = np.float32(np.random.random() < p)
                a = mask * rand_action + (1-mask) * action
                obs, _, _, _ = env.step(a)
                observations.append(obs['observation'])
                actions.append(a)
            obss.append(np.stack(observations, axis=1))
            acts.append(np.stack(actions, axis=1))

        obss = np.concatenate(obss)
        acts = np.concatenate(acts)
        print(obss.shape, acts.shape)

        path_name = os.path.join(path, f'{idx}.pkl')
        with open(path_name, 'wb') as f:
            print('saving... ', path_name)
            pickle.dump([obss, acts], f)


def make_dataset(env_name):
    path = os.path.join('/dataset/', env_name)
    if env_name.startswith('diff_acrobat'):
        from robot.envs.diff_phys.acrobat import GoalAcrobat, IKController
        env = GoalAcrobat(batch_size=200)
        policy = IKController(env)

        if not os.path.exists(path):
            os.makedirs(path)
        batch_collector(env, policy, path, num_episode=25)
        return

    from robot.model.arm.envs import make
    from robot.model.arm.envs.arm_controller import RandomController, Controller as ArmController
    from robot.model.arm.envs.test_acrobat import AcrobatController

    Controller = {
        'arm': ArmController,
        'acrobat2': AcrobatController,
        'plane': RandomController
    }[env_name]

    TIMESPTE = {
        'arm': 100,
        'acrobat2': 50,
        'plane': 50,
    }

    from multiprocessing import Process
    workers = []

    if not os.path.exists(path):
        os.makedirs(path)

    for i in range(20):
        if env_name in ['arm', 'acrobat2']:
            if i >= 10 and i<15:
                policy = lambda x: RandomController(x)
            elif i >= 15:
                policy = lambda x: Controller(x, 0)
            else:
                policy = lambda x, i=i: Controller(x, float(i+1)/11)
        else:
            policy = Controller

        workers.append(Process(
            target=data_collector,
            args=(env_name, policy, 5000//(TIMESPTE[env_name]//50), TIMESPTE[env_name],
                  os.path.join(path, f"{i}.pkl"), make, i==0, i)
        ))

    for i in workers:
        i.start()
    for i in workers:
        i.join()


class Dataset:
    def __init__(self, path, make_dataset=make_dataset, valid_ratio=0.2, device='cuda:0', env=None):
        files = glob.glob(os.path.join(path, '*.pkl'))
        if len(files) == 0:
            env_name = path.split('/')[-1]
            make_dataset(env_name)

        files = glob.glob(os.path.join(path, '*.pkl'))
        observations = []
        actions = []
        idx = []
        for i in files:
            b = int(i.split('/')[-1].split('.')[0])
            idx.append(b)
        idx = sorted(idx)
        for i in idx:
            i = f'{i}.pkl'
            print('load Dataset', i)
            with open(os.path.join(path, i), 'rb') as f:
                data = pickle.load(f)
                observations.append(data[0])
                actions.append(data[1])

        self.obs = np.concatenate(observations)

        self.action = np.concatenate(actions).clip(-1, 1)
        self.device = device

        self.not_inf_mask = (1-(np.isnan(self.obs).sum(axis=(-1, -2)) > 0)) > 0.5
        self.obs = self.obs[self.not_inf_mask]
        self.action = self.action[self.not_inf_mask]
        print('obs max:', np.abs(self.obs).max(axis=(0,1)))
        print('action max:', np.abs(self.action).max(axis=(0,1)))

        idx = np.arange(len(self.obs))

        np.random.seed(0)
        self._rand_idx = np.random.permutation(idx)
        self.obs = self.obs[self._rand_idx]
        self.action = self.action[self._rand_idx]

        self.num_train = int(len(self.obs) * (1-valid_ratio))
        print("num train", self.num_train)
        print("num valid", len(self.obs) - self.num_train)

    def sample(self, mode='train', batch_size=256, timestep=2):
        if mode == 'train':
            idx = np.random.choice(self.num_train, batch_size)
        elif mode == 'valid':
            idx = np.random.choice(len(self.obs) - self.num_train, batch_size) + self.num_train
        else:
            raise NotImplementedError

        idx2 = np.random.randint(self.obs.shape[-2]-timestep, size=batch_size)[:, None] + np.arange(timestep)[None,:]
        s = np.take_along_axis(self.obs[idx], idx2[:,:, None], axis=1)
        a = np.take_along_axis(self.action[idx], idx2[:,:-1, None], axis=1)
        out = [torch.tensor(i, dtype=torch.float, device=self.device) for i in [s, a]]
        return out


def test():
    from robot.model.arm.envs import make
    env_name = 'arm'
    dataset = Dataset('/dataset/{}'.format(env_name))
    # test acrobat
    s = dataset.sample(batch_size=1, timestep=100)[0][0].detach().cpu().numpy()
    from robot.utils import write_video

    env = make(env_name)
    env.reset()

    def make():
        for i in s:
            img = env.render_obs({'observation':i})
            yield img
    write_video(make(), 'xxx.avi')


if __name__ == '__main__':
    test()
