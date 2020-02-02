import gym
from robot.controller.her.ddpg_agent import DDPGAgent
from robot.envs.sapien.exp.utils import RLRecorder

def main():
    #env = gym.make()
    make =gym.make
    timestep = 50
    n_batch = 40
    env_name = 'FetchReach-v1'

    env = make(env_name)
    recorder = RLRecorder(env, '/tmp/tmp/fetchpush', save_model=slice(100000000, None, 1), network_loss=slice(0, None, 50),
                          evaluate=slice(0, None, 50), save_video=1, max_timestep=timestep)
    DDPGAgent(
        n=1, num_epoch=2500, timestep=timestep, n_rollout=2, n_batch=n_batch,
        make=make, env_name=env_name, noise_eps=0.2, random_eps=0.3,
        batch_size=256, future_K=4,
        gamma=0.99, lr=0.001, tau=0.05, update_target_period=1, clip_critic=True,
        device='cpu', recorder=recorder,
    ) # 50

if __name__== '__main__':
    main()