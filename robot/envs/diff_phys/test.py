import gym
import torch
import tqdm
import numpy as np
import cv2
from robot.envs.diff_phys.engine import Articulation2D

def test_acrobat():
    from robot.envs.diff_phys.acrobat_gt import AcrobotEnv
    env = AcrobotEnv()
    #env = gym.make('Acrobot-v1')
    env.reset()
    env.state = np.zeros((2,))
    img = env.render(mode='rgb_array')
    cv2.imshow('x.jpg', img)
    exit(0)
    #cv2.waitKey(0)
    for i in tqdm.trange(10000):
        action = env.action_space.sample()
        img = env.render()
        env.step(action)


def test_articulation():
    articulator = Articulation2D()

    M01 = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, -0.5],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    # v = -w x q
    # we choose q = [0, -0.5, 0] =>
    w, q = [0, 0, 1], [0, 0.5, 0]
    screw1 = w  + (-np.cross(w, q)).tolist()
    # m = 1
    G1 = np.diag([1, 1, 1, 1, 1, 1])
    link1 = articulator.add_link(M01, screw1)
    link1.set_inertial(np.array(G1))
    link1.add_box_visual([0, 0, 0], [0.5, 0.1, 0], (0, 0.8, 0.8))
    link1.add_circle_visual((0, 0, 0), 0.1, (0.8, 0.8, 0.))


    M12 = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, -1.0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    screw2 = screw1
    # m = 1
    G2 = np.diag([1, 1, 1, 1, 1, 1])
    link2 = articulator.add_link(M12, screw2)
    link2.set_inertial(np.array(G2))
    link2.add_box_visual([0, 0, 0], [0.5, 0.1, 0], (0, 0.8, 0.8))
    link2.add_circle_visual((0, 0, 0), 0.1, (0.8, 0.8, 0.))

    EE = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, -0.5],
            [0, 0, 1, 0.],
            [0, 0, 0, 1.],
        ]
    )
    articulator.set_ee(EE)
    articulator.build()

    print(articulator.forward_kinematics(torch.tensor([np.pi/2, np.pi/2], dtype=torch.float64, device='cuda:0')))



if __name__ == '__main__':
    #test_acrobat()
    test_articulation()
