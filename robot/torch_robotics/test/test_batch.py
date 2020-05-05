# the goal is to check the implementation of the batch
import torch
from robot import tr
import tqdm
from robot.torch_robotics import Engine

def make_env(batch_size=1):
    from robot.torch_robotics.contact.elastic import ElasticImpulse
    model = ElasticImpulse(alpha0=0, contact_dof=3, mu=0.3)

    engine = Engine(dt=0.01, frameskip=10, contact_model=model)
    ground = engine.ground(ground_size=20)

    center = tr.togpu([0, 0, 5])[None, :].expand(batch_size, -1)
    inertia = tr.togpu([0.001, 0.001, 0.001])[None, :].expand(batch_size, -1)
    mass = tr.togpu([1]).expand(batch_size)
    radius = tr.togpu([1]).expand(batch_size)
    sphere = engine.sphere(center, inertia, mass, radius, (0, 255, 0), name='sphere')

    sphere2 = engine.sphere(tr.togpu([0, 3, 5])[None, :].expand(batch_size, -1),
                            inertia, mass, radius, (0, 255, 0), name='sphere2')

    renderer = engine.renderer
    renderer.axis(renderer.identity())
    renderer.set_camera_position(-15, 0, 0)
    renderer.set_camera_rotation(0, 0)


    import numpy as np
    sphere.obj.cmass[:, :3, 3] = tr.togpu([0, -3, 1])
    sphere.obj.cmass[:, :3, :3] = tr.projectSO3(tr.togpu(np.random.random(size=(3, 3))))

    sphere2.obj.cmass[:, :3, 3] = tr.togpu([0, 3-0.00001, 1]) #TODO: LCP can't solve break away
    sphere2.obj.cmass[:, :3, :3] = tr.projectSO3(tr.togpu(np.random.random(size=(3, 3))))

    sphere3 = engine.sphere(tr.togpu([0, 5, 1])[None, :].expand(batch_size, -1),
                            inertia, mass, radius, (0, 255, 0), name='sphere3')
    sphere.obj.velocity[:] = tr.togpu([0, 0, 0, 0, 3, 0])
    sphere.obj.velocity = tr.dot(tr.Adjoint(tr.inv_trans(sphere.obj.cmass)), sphere.obj.velocity)
    sphere2.obj.velocity[:] = tr.togpu([0, 0, 0, 0, 0, 0])
    sphere3.obj.velocity[:] = tr.togpu([0, 0, 0, 0, 0, 0])
    return engine, [sphere, sphere2, sphere3]

def test_batch():
    torch.manual_seed(0)
    engine, spheres = make_env(batch_size=30)
    spheres[0].obj.velocity[:] = 0
    spheres[0].obj.velocity[:, 4] = torch.rand_like(spheres[0].obj.velocity[:, 4]) * 5
    spheres[0].obj.velocity = tr.dot(tr.Adjoint(tr.inv_trans(spheres[0].obj.cmass)), spheres[0].obj.velocity)
    cmass, velocity = [i.obj.cmass.clone() for i in spheres], [i.obj.velocity.clone() for i in spheres]
    kk = spheres[0].obj.velocity.clone()
    for i in tqdm.trange(30):
        engine.step()

    for j in range(min(kk.shape[0], 5)):
        torch.manual_seed(0)
        engine2, spheres2 = make_env(batch_size=1)
        for a, c, v in zip(spheres2, cmass, velocity):
            a.obj.cmass[0] = c[j]
            a.obj.velocity[0] = v[j]

        for i in tqdm.trange(30):
            engine2.step()
        for a, b in zip(spheres, spheres2):
            print((a.obj.cmass[j] - b.obj.cmass[0]).abs().max())

if __name__ == '__main__':
    test_batch()
