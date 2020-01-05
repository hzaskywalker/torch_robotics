# run poplin algorithm
from robot.envs.gym import make
from robot.model.gt_model import GTModel
from robot.model.ensemble_nn import EnBNNAgent
from robot.controller.rollout_controller import RolloutCEM
from robot.controller.poplin import PoplinController
from robot.controller.mb_controller import MBController
from robot.utils import Visualizer

def main():
    #env_name = 'MBRLHalfCheetah-v0'
    env_name = 'MBRLCartpole-v0'
    env = make(env_name)
    state_prior = env.state_prior

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

    controller = PoplinController(
        model=model,
        prior=state_prior,
        horizon=30,
        inp_dim=env.observation_space.shape[0],
        oup_dim=env.action_space.shape[0],
        iter_num=5,
        num_mutation=500,
        num_elite=50,
        trunc_norm=True,
    )

    mb_controller = MBController(
        model, controller, timestep=state_prior.TASK_HORIZON,
        path='/tmp/poplin_cartpole',
        batch_size=32,
        valid_ratio=0.1,
    )

    #mb_controller.test(env, print_reward=True, use_tqdm=True)

    mb_controller.init(env)
    for it in range(200):
        print(it, mb_controller.fit(env, progress_buffer_update=False, progress_rollout=True,
                                    progress_train=True, num_train=5))

if __name__ == '__main__':
    main()