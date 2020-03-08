# robot arm model
# we will use envs.hyrule.rl_env.ArmReachGoal to generate enough trianing data...
import os
import glob
import pickle
import tqdm
import numpy as np

# easier for parallel
def data_collector(env, make_policy, num_episode, timestep, path, make=None, use_tqdm=False, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if isinstance(env, str):
        env = make(env)
    policy = make_policy(env)
    ran = tqdm.trange if use_tqdm else range

    observations = actions = geoms = None

    for i in ran(num_episode):
        obs = env.reset()
        for j in range(timestep):
            action = policy(obs)
            if isinstance(obs, dict):
                geom = obs['geom']
                obs = obs['observation']

            if observations is None:
                observations = np.empty([num_episode, timestep + 1, obs.shape[-1]])
                actions = np.empty([num_episode, timestep, action.shape[-1]])
                geoms = np.empty([num_episode, timestep, geom.shape[-1]])

            observations[i, j] = obs
            actions[i, j] = action
            geoms[i, j] = geom

            obs, reward, done, _ = env.step(action)
            if done:
                break

    with open(path, 'wb') as f:
        print('saving... ', path)
        pickle.dump([observations, actions, geoms], f)


def make_dataset(path, env=None):
    if path == '/dataset/arm' or path == '/dataset/arm_with_geom':
        from robot.model.arm.controller import RandomController, Controller
        from robot.envs.hyrule.rl_env import ArmReachWithXYZ
        #if env is None:
        #    env = ArmReachWithXYZ()
        #data_collector(env, lambda x: RandomController(x),
        #               num_episode=1000, timestep=50, path=os.path.join(path, '1.pkl'), use_tqdm=True)
        from multiprocessing import Process
        workers = []
        for i in range(20):
            if i >= 10:
                policy = lambda x: RandomController(x)
            elif i >= 15:
                policy = lambda x: Controller(x, 0)
            else:
                policy = lambda x, i=i: Controller(x, float(i+1)/11)
            make = lambda x: ArmReachWithXYZ(geom=True)
            workers.append(Process(
                target=data_collector,
                args=("not a env", policy, 5000, 50, os.path.join(path, f"{i}.pkl"), make, i==19, i)
            ))
        for i in workers:
            i.start()
        for i in workers:
            i.join()
    else:
        raise NotImplementedError


class Dataset:
    def __init__(self, path, make_dataset=make_dataset, valid_ratio=0.2, env=None):
        files = glob.glob(os.path.join(path, '*.pkl'))
        if len(files) == 0:
            make_dataset(path, env)
        files = glob.glob(os.path.join(path, '*.pkl'))
        observations = []
        actions = []
        geoms = []
        for i in files:
            with open(os.path.join(path, i), 'rb') as f:
                obs, action, geom = pickle.load(f)
                observations.append(obs)
                actions.append(action)
                geoms.append(geom)

        self.obs = np.concatenate(observations)
        self.action = np.concatenate(actions)
        self.geoms = np.concatenate(geoms)
        idx = np.arange(len(self.obs))

        np.random.seed(0)
        idx = np.random.permutation(idx)
        self.obs = self.obs[idx]
        self.action = self.action[idx]
        self.geoms = self.geoms[idx]

        self.num_train = int(len(self.obs) * (1-valid_ratio))
        print("num train", self.num_train)
        print("num valid", len(self.obs) - self.num_train)

    def sample(self, mode='train', batch_size=256, timestep=2, use_geom=False):
        if mode == 'train':
            idx = np.random.choice(self.num_train, batch_size)
        elif mode == 'valid':
            idx = np.random.choice(len(self.obs) - self.num_train, batch_size) + self.num_train
        else:
            raise NotImplementedError

        idx2 = np.random.randint(self.obs.shape[-1]-timestep, size=batch_size)[:, None] + np.arange(timestep)[None,:]
        s = np.take_along_axis(self.obs[idx], idx2[:,:, None], axis=1)
        a = np.take_along_axis(self.action[idx], idx2[:,:-1, None], axis=1)

        if use_geom:
            geom = np.take_along_axis(self.geoms[idx], idx2[:,:, None], axis=1)
            return s, a, geom
        else:
            return s, a


if __name__ == '__main__':
    dataset = Dataset('/dataset/arm_with_geom')
    for i in tqdm.trange(10000):
        x = dataset.sample(batch_size=256)
