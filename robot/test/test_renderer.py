import tqdm
import sys
import numpy as np
from robot.renderer import Renderer
import trimesh
import pickle
import cv2

def test_sphere():
    from robot import renderer
    r = renderer.Renderer()

    sphere = r.sphere((0, 0, 0), 3, (255, 255, 255))
    r.add_point_light((0, 5, 0), color=(255, 0, 0))
    r.set_camera_position(-10, 0, 0)
    from robot import A
    env = A.train_utils.make('acrobat2')

    for i in range(20):
        pos = np.random.random((3,))
        sphere.set_pose(r.translate(pos))
        img = r.render(mode='rgb_array')

    r.sphere((3, 0, 0), 1, (255, 255,255))

    for i in range(30):
        r.render('human')
    img = r.render(mode='rgb_array')
    #env.render()


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
    r = Acrobat2Render('/dataset/acrobatrenderer')
    r.axis(np.eye(4), scale=1.)
    r.set_camera_rotation(1.57, -1.57)

    r.render(mode)

    for i in range(20):
        q = np.random.random((2,)) * np.pi * 2
        r.get('arm').set_pose(q)
        img = r.render(mode)
        if mode == 'rgb_array':
            cv2.imshow('x', img)
            cv2.waitKey(0)


def test_screw():
    from robot.renderer.examples.arm_render import ArmreachRenderer
    mode = sys.argv[1]
    #r = ArmreachRenderer('/dataset/armreachrenderer')
    r = ArmreachRenderer(path=None)

    arm = r.get('arm')
    screw = r.screw_arm(arm.M[0].detach().cpu().numpy(), arm.A[0].detach().cpu().numpy(), name='screw')

    r.set_camera_position(1.5, -0.5, 1.)

    #a = np.random.random((7,)) * np.pi * 2
    a = np.zeros((7,))
    #b = np.random.random((7,)) * np.pi * 2
    for i in range(7):
        b = a.copy()
        k = -np.pi / 3
        a[i] = k
        #b[3] = np.random.random() * np.pi * 2
        b[i] = np.pi *2 + k

        for j in range(1):
            for k in range(20):
                q = a + (b-a)/20 * k
                arm.set_pose(q)
                screw.set_pose(q)
                img = r.render(mode)
                if mode == 'rgb_array':
                    cv2.imshow('x', img)
                    cv2.waitKey(0)
        a[i] = np.pi * 2 + k
    cv2.imwrite('1.jpg', r.render(mode='rgb_array'))


def test_cylinder():
    r = Renderer()
    r.set_camera_position(-4, 0, 0,)
    r.axis(r.identity(), scale=2.)

    r.cylinder(1., 0.3, (255, 255, 255), r.identity())
    r.render('interactive')


if __name__:
    #test_sphere()
    #test_arm()
    #test_two_renders()
    #test_load_render()
    #test_acrobat2_render()
    test_screw()
    #test_cylinder()