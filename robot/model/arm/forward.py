## this file contains all the tools to learn a forward model
## the model should have the following properties:
##      1. rollout
##      2. predict
##      3. update
##      4. update_normalizer(mode='obs') if necessary...

## we will do the following testing: rendering the trajectory prediction video
##    1. testing the model based control performance with the env...
##    2. do evaluate the rollout result in multisteps ...
##    3. render the image

# Wrapper of the model and model based controller
import numpy as np
import torch
import tqdm
from robot.utils import AgentBase, tocpu
from robot.controller.pets.planner import RolloutCEM
from robot.controller.pets.model import EnBNNAgent


class Worker:
    def __init__(self, env, model, dataset, batch_size=256,
                 num_train=1000, num_valid=200,
                 iter_num=5, horizon=20, num_mutation=500, num_elite = 50,
                 traj_length=2, recorder=None, **kwargs):
        assert isinstance(model, AgentBase)
        self.env = env
        self.model: EnBNNAgent = model
        self.controller = RolloutCEM(self.model, self.env.action_space,
                                     iter_num=iter_num, horizon=horizon, num_mutation=num_mutation,
                                     num_elite=num_elite, device=self.model.device, **kwargs)

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_train = num_train
        self.num_valid = num_valid
        self.device = self.model.device
        self.traj_length = traj_length

        self.recoder = recorder

    def select_action(self, x, goal):
        x = torch.tensor(x, dtype=torch.float, device=self.device)
        goal = torch.tensor(goal, dtype=torch.float, device=self.device)
        out = tocpu(self.controller(x, goal))
        return out

    def reset(self):
        self.controller.reset()

    def __call__(self, observation):
        return self.select_action(observation['observation'], observation['desired_goal'])

    def epoch(self, num_train=None, num_valid=None, use_tqdm=False):
        num_train = self.num_train if num_train is None else num_train
        num_valid = self.num_valid if num_valid is None else num_valid

        ran = tqdm.trange if use_tqdm else range

        # train
        for _ in ran(num_train):
            obs, actions = [torch.tensor(i, dtype=torch.float, device=self.device)
                            for i in self.dataset.sample(batch_size=self.batch_size, timestep=self.traj_length)]
            self.model.update_normalizer(obs, 'obs')
            self.model.update_normalizer(actions, 'action')
            info = self.model.update(obs, actions)
            self.recoder.step(self, 1, [info])

        # evaluate
        self.model.eval()
        assert not self.model.training
        valid_output = []
        for _ in ran(num_valid):
            obs, actions = [torch.tensor(i, dtype=torch.float, device=self.device)
                            for i in self.dataset.sample(mode='valid', batch_size=self.batch_size, timestep=self.traj_length)]
            info = self.model.update(obs, actions)
            valid_output.append(info)
        self.recoder.step_eval(valid_output)
        self.model.train()
