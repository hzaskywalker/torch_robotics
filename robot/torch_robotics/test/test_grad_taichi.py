from robot import tr
import cv2
import torch
import tqdm
from robot.torch_robotics import Engine
from robot.torch_robotics.contact.elastic import ElasticImpulse
import numpy as np

def make_env(batch_size=256, gravity=np.array([0, 0, -9.8]), contact_dof=1, dt=0.001, frameskip=10):
    model = ElasticImpulse(alpha0=0, contact_dof=contact_dof)
    engine = Engine(dt=dt, frameskip=frameskip, gravity=tr.togpu(gravity), contact_model=model)
    ground = engine.ground()

    center = tr.togpu([0, 0, 1])[None, :].expand(batch_size, -1)
    inertia = tr.togpu([0.001, 0.001, 0.001])[None, :].expand(batch_size, -1)
    mass = tr.togpu([1]).expand(batch_size)
    radius = tr.togpu([1]).expand(batch_size)
    sphere = engine.sphere(center, inertia, mass, radius, (0, 255, 0), name='sphere')

    renderer = engine.renderer
    renderer.axis(renderer.identity())
    renderer.set_camera_position(-10, 0, 0)
    renderer.set_camera_rotation(0, 0)
    return engine, sphere

def test_taichi():
    # taichi like methods...
    # I am not sure if we should add the continuous collision check into the code ...
    # and I don't know if this is trivial
    batch_size = 30
    engine, sphere = make_env(batch_size=batch_size, contact_dof=1, dt=1, frameskip=1, gravity=np.array([0, 0, 0]))

    value = 2 + torch.arange(batch_size, device='cuda:0').double()/10
    value.requires_grad =True

    sphere.obj.cmass[:, 2, 3] = value[:batch_size]
    sphere.obj.velocity[:, -1] = -2

    print(sphere.obj.cmass[:, 2, 3])
    img = engine.render(mode='rgb_array', render_idx=0)
    cv2.imshow('x', img)
    cv2.waitKey(0)

    for i in tqdm.trange(4):
        engine.step()

        print(sphere.obj.cmass[:, 2, 3])
        img = engine.render(mode='rgb_array', render_idx=0)
        cv2.imshow('x', img)
        cv2.waitKey(0)

    out = sphere.obj.cmass[:, 2, 3]
    print(out)
    out.sum().backward()
    print(value.grad)


if __name__ == "__main__":
    test_taichi()