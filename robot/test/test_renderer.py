import tqdm
import sys
import numpy as np
import trimesh
import pickle
import cv2

def test_sphere():
    from robot import renderer
    r = renderer.Renderer()

    sphere = r.make_sphere((0, 0, 0), 3, (255, 255, 255))
    r.add_point_light((0, 5, 0), color=(255, 0, 0))
    r.set_camera_position(-10, 0, 0)

    pos = np.random.random((3,))
    sphere.set_center(pos)
    img = r.render(mode='human')


def test_arm():
    from robot.renderer.examples.arm_reach_render import ArmreachRenderer

    r = ArmreachRenderer('/dataset/armrender')
    arm = r.get('arm')

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

    sphere = r.make_sphere((0, 0, 0), 3, (255, 255, 255))
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

if __name__:
    #test_sphere()
    test_arm()
    #test_two_renders()
    #test_load_render()