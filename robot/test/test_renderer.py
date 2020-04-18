import tqdm
import sys
import numpy as np
import trimesh
import pickle
import cv2

def test_sphere():
    from robot import renderer
    r = renderer.Renderer()

    sphere = r.sphere((0, 0, 0), 3, (255, 255, 255))
    r.add_point_light((0, 5, 0), color=(255, 0, 0))
    r.set_camera_position(-10, 0, 0)

    pos = np.random.random((3,))
    sphere.set_center(pos)
    img = r.render(mode='human')


def test_arm():
    from robot.renderer.examples.arm_render import ArmreachRenderer

    mode = sys.argv[1]
    r = ArmreachRenderer('/dataset/armrender')
    arm = r.get('arm')
    q = np.random.random((7,)) * np.pi * 2
    arm.set_pose(q)
    r.render(mode=mode)
    exit(0)

    def work():
        for i in range(24):
            q = np.random.random((7,))* np.pi * 2
            arm.set_pose(q)
            img = r.render()
            yield img
    from robot import U
    U.write_video(work(), 'video0.avi')


def test_two_renders():
    from robot import renderer
    r = renderer.Renderer()
    #r2 = renderer.Renderer()

    sphere = r.sphere((0, 0, 0), 3, (255, 255, 255))
    r.add_point_light((0, 5, 0), color=(255, 0, 0))
    r.set_camera_position(-10, 0, 0)

    #sphere = r2.make_sphere((0, 0, 0), 3, (255, 255, 255))
    #r2.add_point_light((0, 5, 0), color=(255, 0, 0))
    #r2.set_camera_position(-10, 0, 0)

    img = r.render()
    #img2 = r2.render()
    img2 = img
    img = np.concatenate((img, img2), axis=1)
    cv2.imwrite('x.jpg', img)

def test_load_render():
    from robot.renderer import Renderer
    r = Renderer.load('xxx.pkl')
    img = r.render()
    cv2.imshow('x', img)
    cv2.waitKey(0)

def test_acrobat2_render():
    from robot.renderer.examples.acrobat2_render import Acrobat2Render
    mode = sys.argv[1]
    r = Acrobat2Render('/dataset/acrobatrenderer', mode)
    r.render(mode)

    for i in range(100):
        q = np.random.random((2,)) * np.pi * 2
        r.get('arm').set_pose(q)
        img = r.render(mode)
        cv2.imshow('x', img)
        cv2.waitKey(0)


def test_screw():
    from robot.renderer.examples.arm_render import ArmreachRenderer
    mode = sys.argv[1]
    r = ArmreachRenderer('/dataset/armreachrenderer')

    arm = r.get('arm')
    screw = r.screw_arm(arm.M[0].detach().cpu().numpy(), arm.A[0].detach().cpu().numpy(), name='screw')

    if mode == 'human' or mode == 'interactive':
        for i in range(20):
            q = np.random.random((7,)) * np.pi * 2
            arm.set_pose(q)
            screw.set_pose(q)
            r.render(mode)
    else:
        for i in range(20):
            q = np.random.random((7,)) * np.pi * 2
            r.get('arm').set_pose(q)
            screw.set_pose(q)
            img = r.render(mode)
            cv2.imshow('x', img)
            cv2.waitKey(1)


if __name__:
    #test_sphere()
    #test_arm()
    #test_two_renders()
    #test_load_render()
    #test_acrobat2_render()
    test_screw()