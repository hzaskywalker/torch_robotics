import os
import numpy as np
from robot.utils.trainer import on_time, merge_training_output
from robot.utils.rl_utils import eval_policy, RLRecorder


def gen_video(env, agent, horizon=24):
    import torch
    # write the video at the neighborhood of the optimal [random] policy
    start = obs = env.reset()
    real_trajs = []
    actions = []
    # 100 frame
    for i in range(horizon):
        action = agent(obs)
        actions.append(action)
        real_trajs.append(env.unwrapped.render_state(obs))
        obs = env.step(action)[0]

    fake_trajs = []
    obs = start
    for i in range(horizon):
        fake_trajs.append(env.unwrapped.render_state(obs))
        s = torch.tensor(obs['observation'], dtype=torch.float32, device=agent.model.device)
        a = torch.tensor(actions[i], dtype=torch.float32, device=agent.model.device)
        obs['observation'] = agent.model.predict(s[None, None, :], a[None, None, :])[0][0, 0].detach().cpu().numpy()

    for a, b in zip(real_trajs, fake_trajs):
        yield np.concatenate((a, b), axis=1)


class ModelRecorder(RLRecorder):
    def step(self, agent, num_train, train_output=None, **kwargs):
        if train_output is not None:
            self._train_output += train_output

            if on_time(self.episode, self.network_loss) and len(train_output) > 0:
                kwargs = {**merge_training_output(self._train_output), **kwargs}
                self._train_output = []

        if on_time(self.episode, self.evaluate):
            kwargs['reward_eval'] = eval_policy(agent, self.get_env(), eval_episodes=self.eval_episodes,
                                                save_video=self.save_video,
                                                video_path=os.path.join(self.path, "video{}.avi"))
            kwargs['rollout'] = gen_video(self.get_env(), agent, horizon=24)

        if on_time(self.episode, self.save_model):
            import torch
            torch.save(agent, os.path.join(self.path, 'agent'))

        self.episode += 1
        self.step_num += num_train
        self.tb(kwargs, self.step_num)

        return kwargs

    def step_eval(self, eval_output, **kwargs):
        for name, val in merge_training_output(eval_output).items():
            kwargs[f'valid_{name}'] = val
        self.tb(kwargs, self.step_num)
        return kwargs
