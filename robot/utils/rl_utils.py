import gym
import tqdm
import cv2
import os
import numpy as np
from .trainer import merge_training_output, on_time
from .tensorboard import Visualizer

def make(env_name):
    from robot.envs.sapien import make as sapien_make
    if '-' in env_name:
        return gym.make(env_name)
    else:
        return sapien_make(env_name)

def get_state(env):
    return env.state_vector()

def set_state(env, state):
    l = (state.shape[0] + 1)//2
    env.set_state(state[:l], state[l:])
    return env

def eval_policy(policy, env_name, seed=12345, eval_episodes=10, save_video=0, video_path="video{}.avi", use_hidden_state=False, progress_episode=False, timestep=int(1e9), start_state=None, print_state=False):
    if isinstance(env_name, str):
        eval_env = make(env_name)
        eval_env.seed(seed + 100)
    else:
        eval_env = env_name

    avg_reward = 0.
    ran = range if not progress_episode else tqdm.trange
    acc = []
    for episode_id in ran(eval_episodes):
        print(episode_id)
        state, done = eval_env.reset(), False
        if start_state is not None:
            set_state(eval_env, start_state)

        out = None
        if isinstance(policy, object):
            if 'reset' in policy.__dir__():
                policy.reset()

        #while not done:
        cc = 0
        for i in ran(timestep):
            print(i)
            if episode_id < save_video:
                if video_path[-3:] == 'avi':
                    img = eval_env.render(mode='rgb_array')
                    if out is None:
                        out = cv2.VideoWriter(
                            video_path.format(episode_id), cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (img.shape[1], img.shape[0]))
                    out.write(img)
                else:
                    eval_env.render()


            print('xxxxxxxxx')
            if use_hidden_state:
                state = get_state(eval_env)
                if print_state:
                    print(state)
            #print()
            #print(','.join(map(lambda x: f"{x:.6f}", list(state))) )
            #print()
            if isinstance(state, dict): pass
            else: state = np.array(state)
            print('not ok')
            action = policy(state)
            print('ok', action)
            state, reward, done, info = eval_env.step(action)
            avg_reward += reward
            cc += reward
            print('xxxxx', done)
            if done:
                break
        if 'is_success' in info:
            acc.append(info['is_success'])

        if progress_episode:
            print(f'episode {episode_id}:', cc)

        if out is not None:
            out.release()

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    if len(acc) > 0:
        print(f"Evaluation success rate over {eval_episodes} episodes: {np.mean(acc):.3f}")
    print("---------------------------------------")
    return avg_reward


class RLRecorder:
    # used to be called by the rl algorithms to record the things for output..
    # use slice to represent the eval an
    # we should count for epsiode ...
    def __init__(self, env_name, path,
                 save_model,
                 network_loss,
                 evaluate,
                 eval_episodes=10,
                 save_video=0,
                 max_timestep=None,
                 tb=True, make=None):
        self.make = make

        assert isinstance(network_loss, slice) or isinstance(network_loss, int) or network_loss is None
        assert isinstance(evaluate, slice) or isinstance(evaluate, int) or evaluate is None
        assert evaluate is None or env_name is not None

        self.path = path
        self.evaluate = evaluate

        self.eval_episodes = eval_episodes
        assert save_video <= self.eval_episodes

        self.save_video = save_video
        self.save_model = save_model
        self.network_loss = network_loss
        self.max_timestep = max_timestep

        self.env = env_name

        self.tb = None if not tb else Visualizer(path)

        self.reset()

    def reset(self):
        self.episode = 0
        self.step_num = 0
        self._train_output = []

    def get_env(self):
        if isinstance(self.env, str):
            if self.make is None:
                self.env =  make(self.env)
            else:
                self.env = self.make(self.env)
        return self.env

    def step(self, agent, reward, episode_timesteps, train_output=None, **kwargs):
        if train_output is not None:
            self._train_output += train_output

            if on_time(self.episode, self.network_loss) and len(train_output) > 0:
                kwargs = {**merge_training_output(self._train_output), **kwargs}
                self._train_output = []

        if on_time(self.episode, self.evaluate):
            kwargs['reward_eval'] = eval_policy(agent, self.get_env(), eval_episodes=self.eval_episodes, save_video=self.save_video,
                                                video_path=os.path.join(self.path, "video{}.avi"))

        if on_time(self.episode, self.save_model):
            import torch
            torch.save(agent, os.path.join(self.path, 'agent'))

        self.episode += 1
        self.step_num += episode_timesteps
        kwargs['reward'] = reward
        kwargs['episode'] = self.episode

        self.tb(kwargs, self.step_num)

        return kwargs
