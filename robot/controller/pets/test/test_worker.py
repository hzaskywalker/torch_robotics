# synchronize trainer
import numpy as np
import torch
from robot.controller.pets.envs import make
from robot.controller.pets.model import EnBNNAgent

def main():
    env, env_params = make('plane')
    model = EnBNNAgent(0.001, env_params.encode_obs, env_params.add_state, env_params.compute_reward,
                       env_params.inp_dim, len(env.action_space.shape), env_params.oup_dim,
                       weight_decay=[0.000025, 0.00005, 0.000075, 0.000075, 0.0001],
                       var_reg=0.01, ensemble_size=5, num_layers=5, mid_channels=200, npart=20).cuda()

    tmp = env.reset()
    obs, goal = tmp['observation'], tmp['desired_goal']

    obs = torch.tensor(obs, dtype=torch.float, device='cuda:0')
    action = torch.tensor(np.array([env.action_space.sample() for i in range(10)]), dtype=torch.float, device='cuda:0')
    goal = torch.tensor(obs, dtype=torch.float, device='cuda:0')
    reward = model.rollout(obs, action[None, :], goal)
    print(reward)


if __name__ == '__main__':
    main()