import torch

class CEMAgent:
    def __init__(self, env, agent, encode_obs, horizon, iter_num, num_mutation, num_elite, device, **kwargs):
        self.env = env

        from robot.controller.pets.planner import RolloutCEM

        self.agent = agent
        self.device = device

        self.encode_obs = encode_obs
        self.controller = RolloutCEM(self.agent, self.env.action_space,
                                     iter_num=iter_num, horizon=horizon, num_mutation=num_mutation,
                                     num_elite=num_elite, device=device, **kwargs)

    def reset(self):
        self.controller.reset()

    def __call__(self, observation):
        x = observation['observation']
        goal = observation['desired_goal']
        x = self.encode_obs(torch.tensor(x, dtype=torch.float, device=self.device))
        goal = torch.tensor(goal, dtype=torch.float, device=self.device)
        out = tocpu(self.controller(x, goal))
        return out
