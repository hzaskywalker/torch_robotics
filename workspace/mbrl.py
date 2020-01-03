from robot.envs.gym import make
from robot.model.gt_model import GTModel
from robot.model.ensemble_nn import EnBNNAgent
from robot.controller.rollout_controller import RolloutCEM
from robot.controller.mb_controller import MBController

def test_env():
    print('loading...')
    env_name = 'MBRLCartpole-v0'
    env = make(env_name)
    state_prior = env.state_prior

    rollout_model = GTModel(make, env_name, num_process=40)

    controller = RolloutCEM(
        model=rollout_model,
        action_space=env.action_space,
        horizon=25,
        iter_num=5,
        num_mutation=100,
        num_elite=10,
        alpha=0.1,
        trunc_norm=True,
        #lower_bound = env.action_space.low,
        #upper_bound =env.action_space.high,
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
    print('acc:', mb_controller.test(env, use_tqdm=True, print_reward=True))


#def test_cartpole():
#    pass

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
        mid_channels=500
    ).cuda()


    #obs = env.reset()[None,:].repeat(5, axis=0)
    #action = env.action_space.sample()[None,:].repeat(5, axis=0)
    #print('OUTPUT', [i.shape for i in model.get_predict(obs, action)])
    #exit(0)
    """
    import torch
    obs = torch.Tensor(env.reset()).to('cuda:0')
    act = torch.Tensor([[env.action_space.sample() for j in range(10)] for i in range(3)]).to('cuda:0')
    for i in model.rollout(obs, act):
        print(i.shape)
    exit(0)
    """
    controller = RolloutCEM(
        model=model,
        action_space=env.action_space,
        horizon=25,
        iter_num=5,
        num_mutation=100,
        num_elite=10,
        alpha=0.1,
        trunc_norm=True,
    )

    mb_controller = MBController(
        model,
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

    mb_controller.test(env, use_tqdm=True)

if __name__ == '__main__':
    #test_env()
    test_cartpole()
