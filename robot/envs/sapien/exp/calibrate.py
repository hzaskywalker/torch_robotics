import numpy as np
import cv2
import torch
import tqdm
import argparse
from robot.utils.rl_utils import make, get_state, set_state
from robot.controller.cem import CEM


def calibrate(env, env_gt, num_env, timestep, vis=True, write_video=0):
    env = env.unwrapped
    env_gt = env_gt.unwrapped
    print(env.get_qpos().shape, env.get_qvel().shape)
    print(env_gt.state_vector().shape)
    def render():
        a = env.render(mode='rgb_array')
        b = env_gt.render(mode='rgb_array')
        return np.concatenate((a, b), axis=1)
    assert env.action_space.shape == env_gt.action_space.shape


    def play(start, actions):
        set_state(env, start)
        set_state(env_gt, start)

        for j in tqdm.trange(1000):
            a = env_gt.action_space.sample()
            r1 = env.step(a)[1]
            r2 = env_gt.step(a)[1]
            print(r1, r2)
            img = render()
            #if write_video:
            #    yield img
            #else:
            cv2.imshow('x', img)
            cv2.waitKey(0)

    if vis:
        for i in range(10):
            env_gt.reset()
            s = get_state(env_gt)
            s[:2] += np.random.uniform(low=-0.5, high=0.5, size=(2,))
            print(s[0])
            set_state(env, s)
            set_state(env_gt, s)
            img = render()
            cv2.imshow('x', img)
            cv2.waitKey(0)

    starts = []
    for i in range(num_env):
        env.reset()
        starts.append(get_state(env))
    starts = np.array(starts)
    actions = np.array([[env_gt.action_space.sample() for _ in range(timestep)] for i in range(num_env)])

    if not write_video:
        play(starts[0], actions[0])
    else:
        raise NotImplementedError
        from robot.utils import write_video
        write_video(play(starts[0], actions[0]), 'video.mp4')

    def measure(scene, params):
        starts, actions = scene
        if isinstance(params, torch.Tensor):
            params = params.detach().cpu().numpy()
        params = params.reshape(params.shape[0], 2, *env_gt.action_space.shape)
        losses = []
        for a, b in tqdm.tqdm(zip(np.exp(params[:, 0]), params[:, 1]), total=len(params)):
            loss = 0
            for i in tqdm.trange(num_env):
                set_state(env, starts[i])
                set_state(env_gt, starts[i])

                for j in tqdm.trange(timestep):
                    env.step(actions[i, j] * a + b)
                    s = get_state(env)
                    env_gt.step(actions[i, j])
                    t = get_state(env_gt)
                    loss += ((t-s)**2).sum()
            loss/=num_env * timestep
            losses.append(loss)
        return torch.tensor(np.array(losses), dtype=torch.float)

    init = torch.tensor(np.array([0, 0] * env.action_space.shape[0]), dtype=torch.float)
    cem = CEM(measure, 5, 50, 5, std=0.3)
    out = cem((starts, actions), init)
    out = out.reshape(2, -1).detach().cpu().numpy()
    print(np.exp(out[0]), out[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('gt_env_name', type=str)
    parser.add_argument('--num_env', type=int, default=20)
    parser.add_argument('--timestep', type=int, default=40)
    parser.add_argument('--write_video', type=int, default=0)
    args = parser.parse_args()

    env = make(args.env_name)
    env_gt = make(args.gt_env_name)

    print(calibrate(env, env_gt, args.num_env, args.timestep, write_video=args.write_video))


if __name__ == '__main__':
    main()
