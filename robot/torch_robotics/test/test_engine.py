from robot import tr
from robot.torch_robotics import Engine

def test_engine():
    engine = Engine()
    ground = engine.ground()

    center = tr.togpu([0, 0, 1])[None, :]
    inertia = tr.togpu([0.001, 0.001, 0.001])[None, :]
    mass = tr.togpu([1])
    radius = tr.togpu([1])
    sphere = engine.sphere(center, inertia, mass, radius, (0, 255, 0), name='sphere')

    center2 = tr.togpu([0, 3, 3])[None, :]
    sphere2 = engine.sphere(center2, inertia, mass, radius, (0, 255, 0), name='sphere2')

    print(sphere.obj.kinetic())
    exit(0)

    renderer = engine.renderer
    renderer.axis(renderer.identity())
    renderer.set_camera_position(-10, 0, 0)
    renderer.set_camera_rotation(0, 0)

    while True:
        engine.step()
        engine.render()

if __name__ == '__main__':
    test_engine()
