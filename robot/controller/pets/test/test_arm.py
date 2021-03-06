# synchronize trainer
from robot.controller.pets.envs import make
from robot.controller.pets.model import EnBNNAgent
from robot.controller.pets.worker import Worker
from robot.controller.pets.forward_recorder import Recoder as RLRecorder

def main():
    env, env_params = make('armreach')
    timestep = 50

    model = EnBNNAgent(0.001, env_params.encode_obs, env_params.add_state, env_params.compute_reward,
                       env_params.inp_dim, env.action_space.shape[0], env_params.oup_dim,
                       weight_decay=[0.000025, 0.00005, 0.000075, 0.000075, 0.0001],
                       var_reg=0.01, ensemble_size=5, num_layers=5, mid_channels=200, npart=20).cuda()

    recorder = RLRecorder(env, 'tmp_arm', save_model=slice(100000000, None, 1), network_loss=slice(0, None, 1),
                          evaluate=slice(10, None, 10), save_video=1, max_timestep=timestep, eval_episodes=5)

    worker = Worker(env, model, maxlen=int(1e6), timestep=timestep, num_train=5, batch_size=200, iter_num=5, horizon=5,
                    num_mutation=500, num_elite=50, recorder=recorder)

    for i in range(1000):
        worker.epoch(1, use_tqdm=True)


if __name__ == '__main__':
    main()