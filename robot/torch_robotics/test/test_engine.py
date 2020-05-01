from robot import tr
import cv2
from robot.torch_robotics import Engine

def test_simple():
    engine = Engine(dt=0.001, frameskip=10)
    ground = engine.ground()

    center = tr.togpu([0, 0, 1])[None, :]
    inertia = tr.togpu([0.001, 0.001, 0.001])[None, :]
    mass = tr.togpu([1])
    radius = tr.togpu([1])
    sphere = engine.sphere(center, inertia, mass, radius, (0, 255, 0), name='sphere')

    center2 = tr.togpu([0, 3, 3])[None, :]
    sphere2 = engine.sphere(center2, inertia, mass, radius, (0, 255, 0), name='sphere2')


    renderer = engine.renderer
    renderer.axis(renderer.identity())
    renderer.set_camera_position(-10, 0, 0)
    renderer.set_camera_rotation(0, 0)

    sphere.obj.apply(tr.togpu([[0, 0, 0, 0, 10, 0]]))

    import cv2
    print(sphere.obj.energy())
    print(sphere2.obj.energy())
    for i in range(100):
        k = sphere.obj.kinetic()
        p = sphere.obj.potential()

        k2 = sphere2.obj.kinetic()
        p2 = sphere2.obj.potential()
        engine.step()
        engine.render()
        #img = engine.render(mode='rgb_array')
        #cv2.imshow('x', img)
        #cv2.waitKey(1)

    print("y axis should be 5", sphere.obj.cmass)
    print("z axis should be -1.9", sphere2.obj.cmass)
    print(sphere.obj.energy())
    print(sphere2.obj.energy())

def test_collision():
    from robot.torch_robotics.contact.elastic import ElasticImpulse
    model = ElasticImpulse(alpha0=0)

    engine = Engine(dt=0.001, frameskip=100, contact_model=model)
    ground = engine.ground()

    center = tr.togpu([0, 1, 3])[None, :]
    inertia = tr.togpu([0.001, 0.001, 0.001])[None, :]
    mass = tr.togpu([1])
    radius = tr.togpu([1])
    sphere = engine.sphere(center, inertia, mass, radius, (0, 255, 0), name='sphere')

    renderer = engine.renderer
    renderer.axis(renderer.identity())
    renderer.set_camera_position(-10, 0, 0)
    renderer.set_camera_rotation(0, 0)

    if False:
        for i in range(20):
            engine.step()
            engine.render('human')
            print(sphere.obj.energy())

    if False:
        sphere.obj.cmass[0, :3, 3] = tr.togpu([0, 0, 5])
        sphere.obj.velocity[0] = tr.togpu([0, 0, 0, 0, 0, 0])

        for i in range(10):
            engine.step()
            engine.render('human')
            print(sphere.obj.energy())

    if True:
        sphere.obj.cmass[0, :3, 3] = tr.togpu([0, -5, 2])
        sphere.obj.velocity[0] = tr.togpu([0, 0, 0, 0, 2, 0])

        for i in range(30):
            engine.step()
            engine.render('human')
            print(sphere.obj.energy())
        print(sphere.obj.cmass[0, :3, 3], f"should be {-5+2*30*0.1}")


def test_two_sphere():
    from robot.torch_robotics.contact.elastic import ElasticImpulse
    model = ElasticImpulse(alpha0=0)

    engine = Engine(dt=0.01, frameskip=10, contact_model=model)
    ground = engine.ground(ground_size=20)

    center = tr.togpu([0, 0, 5])[None, :]
    inertia = tr.togpu([0.001, 0.001, 0.001])[None, :]
    mass = tr.togpu([1])
    radius = tr.togpu([1])
    sphere = engine.sphere(center, inertia, mass, radius, (0, 255, 0), name='sphere')

    sphere2 = engine.sphere(tr.togpu([0, 3, 5])[None, :], inertia, mass, radius, (0, 255, 0), name='sphere2')

    renderer = engine.renderer
    renderer.axis(renderer.identity())
    renderer.set_camera_position(-15, 0, 0)
    renderer.set_camera_rotation(0, 0)

    if True:
        for i in range(50):
            engine.step()
            engine.render()

    if True:
        sphere.obj.cmass[:, :3, 3] = tr.togpu([0,-2,1])
        sphere2.obj.cmass[:, :3, 3] = tr.togpu([0,2,1])
        sphere.obj.velocity[:] = tr.togpu([0, 0, 0, 0, 3, 0])
        sphere2.obj.velocity[:] = tr.togpu([0, 0, 0, 0, 0, 0])

        for i in range(20):
            engine.step()
            engine.render()
            print(sphere.obj.energy()+sphere2.obj.energy())

    if True:
        sphere.obj.cmass[:, :3, 3] = tr.togpu([0,-2,1])
        sphere2.obj.cmass[:, :3, 3] = tr.togpu([0,2,1])
        sphere.obj.velocity[:] = tr.togpu([0, 0, 0, 0, 3, 0])
        sphere2.obj.velocity[:] = tr.togpu([0, 0, 0, 0, -3, 0])

        for i in range(20):
            engine.step()
            engine.render()
            print(sphere.obj.energy()+sphere2.obj.energy())


    import numpy as np
    sphere.obj.cmass[:, :3, 3] = tr.togpu([0, -3, 1])
    sphere.obj.cmass[:, :3, :3] = tr.projectSO3(tr.togpu(np.random.random(size=(3, 3))))

    sphere2.obj.cmass[:, :3, 3] = tr.togpu([0, 3-0.00001, 1]) #TODO: LCP can't solve break away
    sphere2.obj.cmass[:, :3, :3] = tr.projectSO3(tr.togpu(np.random.random(size=(3, 3))))

    sphere3 = engine.sphere(tr.togpu([0, 5, 1])[None, :], inertia, mass, radius, (0, 255, 0), name='sphere3')
    sphere.obj.velocity[:] = tr.togpu([0, 0, 0, 0, 3, 0])
    sphere.obj.velocity = tr.dot(tr.Adjoint(tr.inv_trans(sphere.obj.cmass)), sphere.obj.velocity)
    sphere2.obj.velocity[:] = tr.togpu([0, 0, 0, 0, 0, 0])
    sphere3.obj.velocity[:] = tr.togpu([0, 0, 0, 0, 0, 0])

    print('newton ball')
    print(sphere.obj.energy() + sphere2.obj.energy() + sphere3.obj.energy())
    for i in range(20):
        engine.step()
        print(sphere.obj.energy() + sphere2.obj.energy() + sphere3.obj.energy())
        #exit(0)
        #img = engine.render(mode='rgb_array')
        #cv2.imshow('x', img)
        #cv2.waitKey(1)
        img = engine.render(mode='human')



def test_friction():
    from robot.torch_robotics.contact.elastic import ElasticImpulse
    model = ElasticImpulse(alpha0=0, restitution=1, contact_dof=3, mu=1)

    engine = Engine(dt=0.01, frameskip=10, contact_model=model)
    ground = engine.ground(ground_size=20)


    renderer = engine.renderer
    renderer.axis(renderer.identity())
    renderer.set_camera_position(-15, 0, 0)
    renderer.set_camera_rotation(0, 0)

    center = tr.togpu([0, 0, 1])[None, :]
    inertia = tr.togpu([0.001, 0.001, 0.001])[None, :]
    mass = tr.togpu([1])
    radius = tr.togpu([1])
    sphere = engine.sphere(center, inertia, mass, radius, (0, 255, 0), name='sphere')

    sphere.obj.velocity[:] = tr.togpu([0, 0, 0, 0, 1, 0])

    while True:
        engine.step()
        engine.render()
        print(sphere.obj.velocity)


if __name__ == '__main__':
    #test_simple()
    #test_collision()
    #test_two_sphere()
    test_friction()