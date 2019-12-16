import cv2
import tqdm
import numpy as np

def play(scene, traj, video_path=None, width=640, height=480, cameras=['default'], info=True, env=None):
    from env.env3D import RobotArmEnv
    if env is None:
        assert scene.shape[0] == 4
        env = RobotArmEnv(obstacles=scene.transpose(1, 0), use_viewer=True)

    out = None
    ran = tqdm.trange if info else range

    collision_num = 0

    if not isinstance(traj, list):
        trajs = [traj]
    else:
        trajs = traj

    for i in ran(trajs[0].shape[-1]):
        imgs = []
        for traj in trajs:
            x, y = env.step(traj[...,i])
            img = env.render(mode='rgb_array', width=width, height=height, cameras=cameras)
            if x or y:
                #img = img[...,::-1]
                img[:,:,2] = np.minimum(img[:,:,2] + 128, 255)
                collision_num += 1
            imgs.append(img)
        if len(imgs) > 1:
            img = np.concatenate(imgs, axis=0)
        else:
            img = imgs[0]

        if video_path is not None:
            if out is None:
                out = cv2.VideoWriter(
                    video_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (img.shape[1], img.shape[0]))
            out.write(img)
        else:
            cv2.imshow('x', img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    if out is not None:
        out.release()
    return collision_num

def interpolate_traj(traj, step_size, num_seg=1000):
    import numpy as np
    from mp.helper import Interpolator, PackedTensor
    import torch
    #traj = togpu(traj)
    traj = torch.Tensor(traj)
    interpolator = Interpolator(step_size, num_seg)
    return interpolator(PackedTensor(traj[None,:]))[0].detach().numpy()