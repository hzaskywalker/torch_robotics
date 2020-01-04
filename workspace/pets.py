import tqdm
from robot.envs.gym import make
from robot.model.gt_model import GTModel
from robot.model.ensemble_nn import EnBNNAgent
from robot.controller.rollout_controller import RolloutCEM
from robot.controller.mb_controller import MBController
from robot.utils import Visualizer

def test_env():
    print('loading...')
    env_name = 'MBRLHalfCheetah-v0'
    env = make(env_name)
    state_prior = env.state_prior

    rollout_model = GTModel(make, env_name, num_process=20)

    controller = RolloutCEM(
        model=rollout_model,
        action_space=env.action_space,
        horizon=25,
        iter_num=5,
        num_mutation=500,
        num_elite=50,
        alpha=0.1,
        trunc_norm=True,
        device='cpu',
        upper_bound = env.action_space.high,
        lower_bound =env.action_space.low,
    )

    mb_controller = MBController(
        rollout_model,
        controller,
        maxlen=int(1e6),
        timestep=state_prior.TASK_HORIZON,
        load=False,
        init_buffer_size=3,
        init_train_step=1,
        path=None,
        data_path=None,
        batch_size=32,
        valid_ratio=0.2,
        iters_per_epoch=500,
        valid_batch_num=1,
        data_sampler='fix'
    )

    mb_controller.init(env)
    acc = mb_controller.test(env, use_tqdm=True, print_reward=True)
    print('#' * 10)
    print('acc:', acc)


def load_parameters(model: EnBNNAgent):
    import torch
    xx = torch.load('/home/hza/handful-of-trials-pytorch/model_cheetah.t')
    idx = 0
    for i in model.forward_model.parameters():
        while xx[idx].shape[-1] == 24: #6:
            idx += 1
        print(i.shape, xx[idx].shape)
        i[:] = xx[idx][:]
        idx += 1
    mu = xx[-4]
    var = xx[-3]

    model.obs_norm.mean[:] = mu[0, :18] # 5
    model.obs_norm.std[:] = var[0, :18]

    model.action_norm.mean[:] = mu[0, -6:]
    model.action_norm.std[:] = var[0, -6:]


def test_cartpole():
    print('loading...')
    env_name = 'MBRLCartpole-v0'
    env = make(env_name)
    state_prior = env.state_prior

    model = EnBNNAgent(
        lr=0.001,
        env=env,
        weight_decay=[0.0001, 0.00025, 0.00025, 0.0005],
        var_reg=0.01,
        ensemble_size=5,
        num_layers=4,
        mid_channels=500,
        normalizer=True,
    ).cuda()

    controller = RolloutCEM(
        model=model,
        action_space=env.action_space,
        horizon=25,
        iter_num=5,
        num_mutation=400,
        num_elite=40,
        alpha=0.1,
        trunc_norm=True,
        upper_bound=env.action_space.high,
        lower_bound=env.action_space.low,
    )

    mb_controller = MBController(
        model,
        controller,
        maxlen=int(1e6),
        timestep=int(state_prior.TASK_HORIZON),
        load=False,
        init_buffer_size=1,
        init_train_step=5,
        path='/tmp/xxx',
        data_path=None,
        batch_size=32,
        valid_ratio=0.1,
        iters_per_epoch=500,
        valid_batch_num=1,
        data_sampler='fix',
        vis = Visualizer('/tmp/xxx/history')
    )
    #mb_controller.test(env, use_tqdm=True, print_reward=True)
    #exit(0)

    mb_controller.init(env)
    for it in range(100):
        print(it, mb_controller.fit(env, progress_buffer_update=False, progress_rollout=True,
                                    progress_train=True, num_train=5))


def test_halfcheetah():
    print('loading...')
    env_name = 'MBRLHalfCheetah-v0'
    env = make(env_name)
    state_prior = env.state_prior
    #print(env.observation_space)
    #print(state_prior.inp_dim, env.action_space)

    model = EnBNNAgent(
        lr=0.001,
        env=env,
        weight_decay=[0.000025, 0.00005, 0.000075, 0.000075, 0.0001],
        var_reg=0.01,
        ensemble_size=5,
        num_layers=5,
        mid_channels=200,
        normalizer=False,
    ).cuda()

    controller = RolloutCEM(
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

    if False:
        load_parameters(model)

        """
        obs = env.reset()
        action = env.action_space.sample()
        import torch

        t = model.get_predict(torch.Tensor(obs).cuda()[None,:], torch.Tensor(action).cuda()[None,:])[0]
        print(t)
        print(obs)
        print(env.step(action)[0])
        exit(0)
        """

    mb_controller = MBController(
        model,
        controller,
        maxlen=int(1e6),
        timestep=int(state_prior.TASK_HORIZON),
        load=False,
        init_buffer_size=1,
        init_train_step=5,
        path='/tmp/halfcheetah',
        data_path=None,
        batch_size=32,
        valid_ratio=0.1,
        iters_per_epoch=500,
        valid_batch_num=1,
        data_sampler='fix',
        vis = Visualizer('/tmp/halfcheetah/history')
    )

    #mb_controller.test(env, use_tqdm=True, print_reward=True)
    #exit(0)

    mb_controller.init(env)
    for it in range(200):
        print(it, mb_controller.fit(env, progress_buffer_update=False, progress_rollout=True,
                                    progress_train=True, num_train=5))

if __name__ == '__main__':
    test_cartpole()
    #test_env()
    #test_halfcheetah()
