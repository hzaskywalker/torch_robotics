# run poplin algorithm
import numpy as np
from robot.envs.gym import make
from robot.model.gt_model import GTModel
from robot.model.ensemble_nn import EnBNNAgent
from robot.controller.rollout_controller import RolloutCEM
from robot.controller.poplin import PoplinController
from robot.controller.mb_controller import MBController
from robot.utils import Visualizer

def load_parameters(model: EnBNNAgent):
    import torch
    from scipy.io import loadmat
    xx = loadmat('/home/hza/handful-of-trials-pytorch/POPLIN/log/POPLINP_AVG/2020-01-05--17:34:45/model.mat')
    idx = 2
    with torch.no_grad():
        for i in model.forward_model.parameters():
            while xx[str(idx)].shape[-1] == 24: #6:
                idx += 1
            i[:] = torch.tensor(xx[str(idx)][:], dtype=torch.float, device='cuda:0')
            idx += 1
        mu = torch.tensor(xx['0'], dtype=torch.float, device='cuda:0')
        var = torch.tensor(xx['1'], dtype=torch.float, device='cuda:0')

        model.obs_norm.mean[:] = mu[0, :18] # 5
        model.obs_norm.std[:] = var[0, :18]

        model.action_norm.mean[:] = mu[0, -6:]
        model.action_norm.std[:] = var[0, -6:]


class MBController2(MBController):
    def __init__(self, *args, **kwargs):
        from scipy.io import loadmat
        xx = loadmat('/home/hza/handful-of-trials-pytorch/POPLIN/log/POPLINP_AVG/2020-01-05--17:34:45/logs.mat')
        print(xx['observations'].shape)
        print(xx['actions'].shape)
        self._observations = xx['observations']
        self._actions = xx['actions']
        self._data_count = 0
        super(MBController2, self).__init__(*args, **kwargs)

    def update_buffer(self, env, policy, num_traj, progress=False, progress_rollout=False):
        s = self._observations[self._data_count]
        a = self._actions[self._data_count]
        a = np.concatenate((a, a[-1:]), axis=0)

        self.buffer.update(s, a)
        self._data_count += 1
        return None

    def init(self, env):
        pass


def main():
    env_name = 'MBRLHalfCheetah-v0'
    #env_name = 'MBRLCartpole-v0'
    env = make(env_name)
    extension = env.extension

    model = EnBNNAgent(
        lr=0.001,
        env=env,
        weight_decay=[0.000025, 0.00005, 0.000075, 0.000075, 0.0001],
        var_reg=0.01,
        ensemble_size=5,
        num_layers=5,
        mid_channels=200,
        normalizer=True,
        npart=20,
    ).cuda()

    #model = GTModel(
    #    make, env_name, num_process=30
    #)
    #load_parameters(model)
    #print(model.obs_norm.mean, model.obs_norm.std)

    #exit(0)
    #import torch
    #model = torch.load('/tmp/poplin_{}/agent'.format(env_name))
    #print(model.obs_norm.mean, model.obs_norm.std)
    #exit(0)
    #print('xxxx')

    controller = PoplinController(
        model=model,
        extension=extension,
        horizon=30, #30,
        inp_dim=extension.inp_dim,
        oup_dim=env.action_space.shape[0],
        iter_num=5,
        num_mutation=500,
        num_elite=50,
        alpha=0.1,
        action_space=env.action_space, # constrain the action low and high
        trunc_norm=True,
        std=0.1 ** 0.5,
        num_layers=2,
    )

    #model.rollout = model.rollout2
    controller2 = RolloutCEM(
        model=model,
        action_space=env.action_space,
        horizon=30,
        iter_num=5,
        num_mutation=500,
        num_elite=50,
        alpha=0.1,
        trunc_norm=True,
        upper_bound=env.action_space.high,
        lower_bound=env.action_space.low,
    )

    mb_controller = MBController(
        model, controller2, timestep=1000, #int(extension.TASK_HORIZON * 0.1),
        path='/tmp/poplin_{}'.format(env_name),
        batch_size=32,
        valid_ratio=0.0,
    )

    #acc = mb_controller.test(env, print_reward=True, use_tqdm=True)
    #print()
    #print(acc)
    #exit(0)

    """
    for i in range(51):
        mb_controller.update_buffer(None, None, 1)
    #s = mb_controller.buffer.make_sampler('fix', 'train', 5)
    mb_controller.model.fit_normalizer(mb_controller.buffer)
    exit(0)

    s = mb_controller.buffer.make_sampler('fix', 'train', 5)
    for i in s:
        for j in i:
            print(j.shape)
        print('xxxxx')

    exit(0)
    """

    mb_controller.init(env)
    for it in range(100):
        print(it, mb_controller.fit(env, progress_buffer_update=False, progress_rollout=True,
                                    progress_train=True, num_train=5))

        print("I am using controller2")

if __name__ == '__main__':
    main()