import torch
from robot.envs.dm_control import make as dm_make
from robot.envs.gym import make
from robot.controller.mb_controller import MBController
from robot.controller.rollout_controller import RolloutCEM

model:MBController = torch.load('/tmp/halfcheetah/agent')

env = make('MBRLHalfCheetah-v0')

agent = RolloutCEM(
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

obs = env.reset()
agent.reset()
for i in range(1000):
    env.render()

    act = agent(obs).detach().cpu().numpy()
    t = env.step(act)[0]
