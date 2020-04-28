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

if __name__ == '__main__':
    #test_simple()
    test_collision()
