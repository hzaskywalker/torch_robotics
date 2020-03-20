from robot.model.arm.train import *
import cv2
env_name = 'plane'
env = make(env_name)
agent = torch.load('rollout/agent')
agent.device = 'cuda:0'

renderer = Renderer(env_name, env)

from robot.model.arm.envs.arm_controller import RandomController

policy = RolloutCEM(agent, env.action_space, iter_num=10, horizon=10 - 1, num_mutation=500,
                        num_elite=50, device=agent.device)
policy.reset()

from robot.utils import write_video
write_video(renderer.render_video(policy, agent, horizon=24), path='xxx.avi')
#for idx, i in enumerate):
#    cv2.imwrite(f'x{idx}.jpg', i)
