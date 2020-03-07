# synchronize trainer
import numpy as np
import torch
from robot.controller.pets.envs import make
from robot.controller.pets.model import EnBNNAgent
from robot.controller.pets.worker import Worker
from robot.controller.pets.forward_recorder import RLRecorder as RLRecorder

def main():
    env, env_params = make('cheetah')
    timestep = 1000

    model = EnBNNAgent(0.001, env_params.encode_obs, env_params.add_state, env_params.compute_reward,
                       env_params.inp_dim, env.action_space.shape[0], env_params.oup_dim,
                       weight_decay=[0.000025, 0.00005, 0.000075, 0.000075, 0.0001],
                       var_reg=0.01, ensemble_size=5, num_layers=5, mid_channels=200, npart=20).cuda()

    recorder = RLRecorder(env, 'tmp', save_model=slice(100000000, None, 1), network_loss=slice(0, None, 1),
                          evaluate=slice(10, None, 10), save_video=1, max_timestep=timestep, eval_episodes=1)

    worker = Worker(env, model, maxlen=int(1e6), timestep=timestep, num_train=5, batch_size=32, iter_num=5, horizon=30,
                    num_mutation=200, num_elite=10, recorder=recorder,
                    alpha=0.1, trunc_norm=True, upper_bound=env.action_space.high, lower_bound=env.action_space.low)

    policy = lambda a, b: env.action_space.sample()
    env.reset()
    obs = env.step(env.action_space.sample())[0]

    for i in range(1000):
        worker.epoch(1, use_tqdm=True, policy=policy)
        policy = None


if __name__ == '__main__':
    main()