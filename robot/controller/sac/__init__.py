from .sac import SAC
from .replay_buffer import ReplayMemory
import itertools


def sac(env, num_steps=1000001, replay_size=1000000, start_steps=10000, batch_size=256,
        updates_per_step=1, recorder=None, **kwargs):
    # Agent
    agent = SAC(env.observation_space.shape[0], env.action_space, **kwargs)

    # Memory
    memory = ReplayMemory(replay_size)

    # Training Loop
    total_numsteps = 0
    updates = 0


    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        train_outputs = []
        while not done:
            if start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > batch_size:
                # Number of updates per step in environment
                for i in range(updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
                                                                                                         batch_size,
                                                                                                         updates)

                    train_outputs.append({
                        'critic_loss': critic_1_loss,
                        'critic_2_loss': critic_2_loss,
                        'policy_loss': policy_loss,
                        'ent_loss': ent_loss,
                        'alpha': alpha,
                    })
                    updates += 1

            next_state, reward, done, _ = env.step(action)  # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            if episode_steps == env._max_episode_steps:
                done = True

            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            memory.push(state, action, reward, next_state, mask)  # Append transition to memory

            state = next_state

        if total_numsteps > num_steps:
            break

        #writer.add_scalar('reward/train', episode_reward, i_episode)
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps,
                                                                                      episode_steps,
                                                                                      round(episode_reward, 2)))
        if recorder is not None:
            recorder.step(agent, episode_reward, episode_steps, train_outputs)

    return agent
