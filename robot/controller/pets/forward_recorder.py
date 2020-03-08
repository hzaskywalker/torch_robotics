# forward recorder that can r the
import numpy as np
from .worker import Worker
from robot.utils.rl_utils import RLRecorder, on_time

class Recoder(RLRecorder):
    def step(self, agent: Worker, reward, episode_timesteps, train_output=None, **kwargs):
        env = self.get_env()
        def gen_video():
            import torch
            # write the video at the neighborhood of the optimal [random] policy
            horizon = 24
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
                obs['observation'] = agent.model.predict(s[None,None,:],a[None,None,:])[0][0,0].detach().cpu().numpy()

            for a, b in zip(real_trajs, fake_trajs):
                yield np.concatenate((a, b), axis=1)

        if on_time(self.episode, self.evaluate):
            kwargs['video_model'] = gen_video()
        super(Recoder, self).step(agent, reward, episode_timesteps, train_output, **kwargs)

