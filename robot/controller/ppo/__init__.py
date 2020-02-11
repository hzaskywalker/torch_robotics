import torch
from .ppo import PPO
from .models import Policy
from .storage import RolloutStorage
from . import utils
from collections import deque


def ppo(envs, recurrent_policy=False, device='cuda:0',
        clip_param=0.2, ppo_epoch=10, num_mini_batch=32,
        value_loss_coef=0.5, entropy_coef=0.01, lr=7e-4, eps=1e-5, max_grad_norm=0.5,
        num_steps=2048, num_processes=1, num_env_steps=1000000, use_linear_lr_decay=True,
        use_gae=True, gamma=0.99, gae_lambda=0.95, use_proper_time_limits=True, recoder=None):
    """
    :param lr: learning rate (default: 7e-4)
    :param eps: RMSprop optimizer epsilon (default: 1e-5)
    :return:
    """
    actor_critic = Policy(envs.observation_space, envs.action_space,
                          base_kwargs={'recurrent': recurrent_policy})
    actor_critic.to(device)


    agent = PPO(
        actor_critic,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        value_loss_coef,
        entropy_coef,
        lr=lr,
        eps=eps,
        max_grad_norm=max_grad_norm)

    rollouts = RolloutStorage(num_steps, num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)


    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    num_updates = int(num_env_steps) // num_steps // num_processes
    for j in range(num_updates):

        if use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates, lr)

        for step in range(num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.tensor([[0.0] if done_ else [1.0] for done_ in done], dtype=torch.float)
            bad_masks = torch.tensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos], dtype=torch.float)
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, use_gae, gamma, gae_lambda, use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        if recoder is not None:
            recoder.step(agent, )
