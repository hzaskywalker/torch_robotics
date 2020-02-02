import argparse
from robot.envs.sapien.exp.utils import make, RLRecorder
from robot.controller.sac import sac

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default=None)
    parser.add_argument('--start_timesteps', type=int, default=10000)
    parser.add_argument('--max_timesteps', type=int, default=1000000)
    parser.add_argument('--replay_size', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--updates_per_step', type=int, default=1)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--policy', type=str, default='Gaussian')
    parser.add_argument('--target_update_interval', type=int, default=1)
    parser.add_argument('--automatic_entropy_tuning', type=int, default=0)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-3)

    parser.add_argument('--path', type=str, default='/tmp/tmp')
    args = parser.parse_args()

    env = make(args.env_name)
    print(env.action_space)
    print(env.observation_space)

    # TODO: note we use the same environment for testing... which is troublesome.
    # TODO: if we can run multiple env together, we can must
    recorder = RLRecorder(env, args.path, save_model=slice(0, None, 10), network_loss=slice(0, None, 10),
                          evaluate=slice(0, None, 10), save_video=1, max_timestep=args.max_timesteps)

    #td3(env, args.seed, args.start_timesteps, args.eval_freq, args.max_timesteps, args.expl_noise, args.batch_size,
    #    args.discount, args.tau, args.policy_noise, args.noise_clip, args.policy_freq, recorder=recorder)
    sac(env, num_steps=args.max_timesteps, replay_size=args.replay_size, start_steps=args.start_timesteps,
        batch_size=args.batch_size, updates_per_step=args.updates_per_step, gamma=args.discount, tau=args.tau,
        alpha=args.alpha, policy=args.policy, target_update_interval=args.target_update_interval,
        automatic_entropy_tuning=args.automatic_entropy_tuning, hidden_size=args.hidden_size, lr=args.lr, recorder=recorder)

if __name__ == '__main__':
    main()