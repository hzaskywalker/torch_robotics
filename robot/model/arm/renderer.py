import numpy as np
import torch

class Renderer:
    # renderer is not the state
    def __init__(self, env_name, env):
        self.env_name = env_name
        self.env = env

    def render_state(self, s, ee=None):
        # s is the predicted state ...
        s = s[:s.shape[-1]//2] # only use the last shape

        if self.env_name == 'arm':
            assert s.shape[0] == 7
            q = np.zeros((29,))
            q[np.arange(7)+1] = s
            if ee is not None:
                q[-3:] = ee
            return self.env.unwrapped.render_obs(
                {'observation': q}, reset=False)
        elif self.env_name == 'acrobat2':
            assert s.shape[0] == 2
            t = (s,)
            if ee is not None:
                t = t + (ee,)
            else:
                t = t+ (np.array((0, 0, 0)),)
            q = np.concatenate(t)
            return self.env.unwrapped.render_obs(
                {'observation': q},reset=False)
        elif self.env_name == 'plane':
            return self.env.unwrapped.render_obs(
                {'observation': s}
            )
        else:
            raise NotImplementedError

    def tocpu(self, a):
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
        return a

    def render_image(self, start, future, ee,
                     predict_future, predict_ee, num=2):
        start = self.tocpu(start)
        future = self.tocpu(future)
        ee = self.tocpu(ee)
        # TODO: how to visualize the velocity?
        images = []
        for i in range(num):
            start_img = self.render_state(start[i])
            ground_truth = [start_img] + [self.render_state(s, ee) for s, ee in zip(future[i], ee[i])]
            predicted = [start_img]+ [self.render_state(s, ee) for s, ee in zip(predict_future[i], predict_ee[i])]
            images.append(np.concatenate((
                np.concatenate(ground_truth, axis=1),
                np.concatenate(predicted, axis=1),
            ), axis=0))
        return np.stack(images)


    def render_video(self, policy, agent, horizon=24):
        env = self.env
        start = obs = env.reset()
        real_trajs = []
        actions = []
        # 100 frame

        if 'reset' in policy.__dict__:
            policy.reset()
        for i in range(horizon):
            action = policy(obs)
            actions.append(action)
            real_trajs.append(env.unwrapped.render_obs(obs))
            obs = env.step(action)[0]

        fake_trajs = []
        obs = start

        # rollout_functiont take obs, actions as input
        device = agent.device
        start_obs = torch.tensor(obs['observation'], dtype=torch.float, device=device)[None,:]
        actions = torch.tensor(np.array(actions), dtype=torch.float, device=device)[None,:]
        state, ee = agent.rollout(start_obs, actions, return_traj=True)

        state = state[0].detach().cpu().numpy()
        ee = ee[0].detach().cpu().numpy()

        fake_trajs.append(env.unwrapped.render_obs(obs))
        for s, e in zip(state, ee):
            fake_trajs.append(self.render_state(s, e))

        for a, b in zip(real_trajs, fake_trajs):
            out = np.concatenate((a, b), axis=1)
            yield out
