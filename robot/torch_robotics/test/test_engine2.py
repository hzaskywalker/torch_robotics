from robot import tr
from robot.torch_robotics.engine2 import Engine2
import tqdm


def test_collision():
    engine = Engine2(contact_dof=1, frameskip=10, mu=1)

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
    engine.rigid_body[:, 0].velocity[:] = tr.togpu([0, 0, 0, 0, 1, 0])

    for i in tqdm.trange(30):
        engine.step()
        engine.render()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    choice = [i[5:] for i in globals() if i.startswith('test_')]
    parser.add_argument('task', type=str, choices=choice)
    args = parser.parse_args()
    eval('test_'+args.task+'()')
