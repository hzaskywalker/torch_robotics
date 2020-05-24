from robot import tr
from robot.torch_robotics.geometry.simplex3 import Simplex


def test():
    sim = Simplex(0)
    center = tr.togpu([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])[None, :]
    size = tr.togpu([1, 1, 1])[None, :]

    box = sim.box(center, size)

    center2 = center.clone()
    center2[:,2,3] = 0.95
    center2[:,1,3] = 0.3

    size = tr.togpu([0.9, 0.9, 0.9])[None, :]
    box2 = sim.box(center2, size)

    shapes = [box, box2]
    collisions = sim.collide(shapes, update=True)

    sim = collisions.sim
    print(sim.normal_pos[:, 0])
    print(sim.normal_pos[:, 1])


if __name__ == '__main__':
    test()