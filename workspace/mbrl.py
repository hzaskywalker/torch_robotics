from robot.envs.gym import make
from robot.model.gt_model import GTModel
from robot.controller.rollout_controller import RolloutCEM
from robot.controller.mb_controller import MBController

def test_env():
    print('loading...')
    env_name = 'MBRLCartpole-v0'
    env = make(env_name)
    state_prior = env.state_prior

    rollout_model = GTModel(make, env_name, num_process=40)

    controller = RolloutCEM(
        rollout=rollout_model,
        action_space=env.action_space,
        horizon=25,
        iter_num=5,
        num_mutation=400,
        num_elite=40,
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

if __name__ == '__main__':
    test_env()