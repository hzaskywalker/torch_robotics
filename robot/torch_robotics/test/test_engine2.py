from robot import tr
from robot.torch_robotics.engine2 import Engine2
import tqdm


def test_collision():
    engine = Engine2(contact_dof=3, frameskip=10, dt=0.01, mu=1)

    center = tr.togpu([0, 0, 1])[None, :]
    inertia = tr.togpu([0.001, 0.001, 0.001])[None, :]
    mass = tr.togpu([1])
    radius = tr.togpu([1])
    sphere = engine.sphere(center, inertia, mass, radius, (0, 255, 0), name='sphere')
    ground = engine.ground(10)

    center2 = tr.togpu([0, 3, 1])[None, :]
    sphere2 = engine.sphere(center2, inertia, mass, radius, (0, 255, 0), name='sphere2')

    center3 = tr.togpu([0, 5.01, 1])[None, :]
    sphere3 = engine.sphere(center3, inertia, mass, radius, (0, 255, 0), name='sphere3')

    engine.add(*sphere).add(*sphere2).add(*sphere3).add(*ground).reset()
    engine.rigid_body[:, 0].velocity[:] = tr.togpu([0, 0, 0, 0, 3, 0])

    for i in tqdm.trange(30):
        engine.step()
        engine.render()


def test_box():
    engine = Engine2(dt=0.001, frameskip=10, collision_threshold=1e-3, mu=1., contact_dof=3, restitution=1)
    ground = engine.ground(10)

    center = tr.togpu([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0.5],
                       [0, 0, 0, 1]])[None, :]
    inertia = tr.togpu([0.001, 0.001, 0.001])[None, :]
    mass = tr.togpu([5])
    size = tr.togpu([1, 1, 1])[None, :]

    box = engine.box(center, inertia, mass, size, (255, 0, 0, 180), 'box1')
    box[0].velocity = tr.togpu([0, 0, 0, 0, 5, 0])[None, :]
    engine.add(*box).add(*ground).reset()
    # N=mg, f = \mu mg
    # v^2 - v_0^2 = 2ax => x = v_0^2/2a

    import tqdm
    for i in tqdm.trange(100):
        engine.render()
        engine.step()
    print(engine.rigid_body[:, 0].cmass[..., :3, :4])


def test_sphere_box():
    engine = Engine2(dt=0.005, frameskip=10, contact_dof=3, mu=1., restitution=0)
    ground = engine.ground(10)

    center = tr.togpu([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0.5],
                       [0, 0, 0, 1]])[None, :]
    inertia = tr.togpu([0.001, 0.001, 0.001])[None, :]
    mass = tr.togpu([0.1])
    size = tr.togpu([1, 1, 1])[None, :]

    box = engine.box(center, inertia, mass, size, (255, 0, 0, 180), 'box1')
    box[0].velocity = tr.togpu([0, 0, 0, 0, 0, 0])[None, :]
    engine.add(*box)


    center = tr.togpu([0, 0.3, 1.2])[None, :]
    inertia = tr.togpu([0.001, 0.001, 0.001])[None, :]
    mass = tr.togpu([5])
    radius = tr.togpu([0.2])
    sphere = engine.sphere(center, inertia, mass, radius, (0, 255, 0), name='sphere')
    sphere[0].velocity[:] = tr.togpu([0, 0, 0, 0, 1, 0])

    center2 = tr.togpu([0, -0.3, 1.2])[None, :]
    sphere2 = engine.sphere(center2, inertia, mass, radius, (0, 255, 0), name='sphere2')
    sphere2[0].velocity[:] = tr.togpu([0, 0, 0, 0, 1, 0])

    engine.add(*ground).add(*sphere).add(*sphere2).reset()

    for i in tqdm.trange(100):
        engine.render()
        engine.step()


def test_two_sphere():
    engine = Engine2(contact_dof=3, frameskip=10, dt=0.01, mu=1)

    center = tr.togpu([0, 0, 1])[None, :]
    inertia = tr.togpu([0.001, 0.001, 0.001])[None, :]
    mass = tr.togpu([1])
    radius = tr.togpu([1])
    sphere = engine.sphere(center, inertia, mass, radius, (0, 255, 0), name='sphere')
    ground = engine.ground(10)

    center2 = tr.togpu([0, 0, 3])[None, :]
    sphere2 = engine.sphere(center2, inertia, mass, radius, (0, 255, 0), name='sphere2')

    engine.add(*sphere).add(*sphere2).add(*ground).reset()
    engine.rigid_body[:, 0].velocity[:] = tr.togpu([0, 0, 0, 0, 1, 0])

    for i in tqdm.trange(30):
        engine.step()
        engine.render()

def test_articulation():
    engine = Engine2(contact_dof=3, frameskip=10, dt=0.01, mu=1)

    articulation = engine.robot('xxx')
    ground = engine.ground(10)
    #engine.add(*ground).add(*articulation).reset()
    engine.add(*articulation).reset()
    for i in range(30):
        engine.render('interactive')
        engine.step()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    choice = [i[5:] for i in globals() if i.startswith('test_')]
    parser.add_argument('task', type=str, choices=choice)
    args = parser.parse_args()
    eval('test_'+args.task+'()')
