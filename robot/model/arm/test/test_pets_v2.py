from robot import A
import sys

def cheetah():
    args = A.train_utils.get_args()
    args.env_name = 'cheetah'
    args.path = 'pets_cheetah'
    args.lr = 0.001
    args.num_train_iter = 5
    A.pets.online_trainer(args)

def arm():
    args = A.train_utils.get_args()
    args.env_name = 'arm'
    args.path = 'pets_arm'
    args.lr = 0.001
    args.num_train_iter = 5
    A.pets.online_trainer(args)

def test_arm():
    import torch
    from robot import U, A
    agent = torch.load('pets_arm/agent')
    env = A.train_utils.make('arm')

    rollout_predictor = A.pets.PetsRollout(agent, A.pets.ArmFrame, npart=20)

    controller = A.train_utils.RolloutCEM(rollout_predictor, env.action_space, iter_num=5, horizon=30,
                                 num_mutation=500, num_elite=50, device='cuda:0', alpha=0.1, trunc_norm=True,
                                 lower_bound=env.action_space.low, upper_bound=env.action_space.high)
    avg_reward, trajectories = U.eval_policy(controller, env, eval_episodes=10, save_video=1, progress_episode=True,
                                             video_path= "video{}.avi", return_trajectories=True,
                                             timestep=100)

if __name__ == '__main__':
    test_arm()
    exit(0)
    args = A.train_utils.get_args()
    if args.env_name == 'cheetah':
        cheetah()
    elif args.env_name == 'arm':
        arm()
