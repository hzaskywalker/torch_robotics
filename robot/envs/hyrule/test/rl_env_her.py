# we should not use reset...
import argparse
from robot.envs.sapien.exp.utils import RLRecorder
from robot.controller.her.ddpg_agent import DDPGAgent
from gym.wrappers import TimeLimit


def make(env_name):
    from robot.envs.hyrule.rl_env import RLEnv
    env = RLEnv(env_name)
    env.reset()
    env = TimeLimit(env, 50)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, required=True)
    parser.add_argument('--path', type=str, required=True) # set this to 0 to avoid use her
    parser.add_argument('--n', type=int, default=8)
    parser.add_argument('--num_epoch', type=int, default=1000000)
    parser.add_argument('--timestep', type=int, default=50)
    parser.add_argument('--n_batch', type=int, default=50)
    parser.add_argument('--noise_eps', type=float, default=0.2)
    parser.add_argument('--random_eps', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--future_K', type=int, default=4) #set this to 0 to avoid use her
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--tau', type=float, default=0.05)
    parser.add_argument('--update_target_period', type=int, default=1)
    parser.add_argument('--clip_critic', type=int, default=1) # set this to 0 to avoid use her
    parser.add_argument('--use_her', type=int, default=1) # set this to 0 to avoid use her
    args = parser.parse_args()

    recorder = RLRecorder(args.env_name, args.path, save_model=slice(100000000, None, 1), network_loss=slice(0, None, 50),
                          evaluate=slice(0, None, 50), save_video=1, max_timestep=args.timestep, make=make)
    if not args.use_her:
        assert not args.clip_critic and args.future_K==0

    DDPGAgent(
        n=args.n, num_epoch=args.num_epoch, timestep=args.timestep, n_rollout=2, n_batch=args.n_batch,
        make=make, env_name=args.env_name, noise_eps=args.noise_eps, random_eps=args.random_eps,
        batch_size=args.batch_size, future_K=args.future_K,
        gamma=args.gamma, lr=args.lr, tau=args.tau, update_target_period=1, clip_critic=args.use_her,
        device='cuda:0', recorder=recorder,
    )



if __name__ == '__main__':
    main()
