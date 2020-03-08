# robot arm model
# we will use envs.hyrule.rl_env.ArmReachGoal to generate enough trianing data...
import os
import glob
import pickle
import tqdm
import numpy as np

# easier for parallel
def data_collector(env, make_policy, num_episode, timestep, path, make=None, use_tqdm=False):
    if isinstance(env, str):
        env = make(env)
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
            if done:
                break

    with open(path, 'wb') as f:
        print('saving... ', path)
        pickle.dump([observations, actions], f)


def make_dataset(path, env=None):
    if path == '/dataset/arm':
        if env is None:
            from robot.envs.hyrule.rl_env import ArmReachWithXYZ
            from robot.model.arm.controller import RandomController
            env = ArmReachWithXYZ()
        data_collector(env, lambda x: RandomController(x),
                       num_episode=1000, timestep=50, path=os.path.join(path, '1.pkl'), use_tqdm=True)
    else:
        raise NotImplementedError


class Dataset:
    def __init__(self, path, make_dataset=make_dataset, valid_ratio=0.2, env=None):
        files = glob.glob(os.path.join(path, '*.pkl'))
        if len(files) == 0:
            make_dataset(path, env)
        observations = []
        actions = []
        for i in files:
            with open(os.path.join(path, i), 'rb') as f:
                obs, action = pickle.load(f)
                observations.append(obs)
                actions.append(action)

        self.obs = np.concatenate(observations)
        self.action = np.concatenate(actions)
        self.num_train = int(len(obs) * (1-valid_ratio))
        print("num train", self.num_train)
        print("num valid", len(obs) - self.num_train)

    def sample(self, mode='train', batch_size=256, timestep=2):
        if mode == 'train':
            idx = np.random.choice(self.num_train, batch_size)
        elif mode == 'valid':
            idx = np.random.choice(len(self.obs) - self.num_train, batch_size) + self.num_train
        else:
            raise NotImplementedError

        idx2 = np.random.randint(self.obs.shape[-1]-timestep, size=batch_size)[:, None] + np.arange(timestep)[None,:]
        s = np.take_along_axis(self.obs[idx], idx2[:,:, None], axis=1)
        a = np.take_along_axis(self.action[idx], idx2[:,:-1, None], axis=1)
        return s, a


if __name__ == '__main__':
    dataset = Dataset('/dataset/arm')
    for i in tqdm.trange(10000):
        x = dataset.sample(batch_size=256)
