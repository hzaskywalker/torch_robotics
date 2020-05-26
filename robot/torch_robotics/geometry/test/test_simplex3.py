from robot import tr
from robot.torch_robotics.geometry.simplex import Simplex


def test():
    sim = Simplex(0, compute_gradients=True)
    center = tr.togpu([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])[None, :]
    size = tr.togpu([1, 1, 1])[None, :]

    center.requires_grad = True
    box = sim.box(center, size)

    center2 = tr.togpu([0, 0, 1.])[None,:]
    center2.requires_grad = True
    sphere = sim.sphere(center2, tr.togpu([0.5]))

    shapes = [box, sphere]
    sim.collide(shapes, update=True)

    (sim.dist.sum()+sim.pose.sum()).backward()
    print(center.grad)
    print(center2.grad)


if __name__ == '__main__':
    test()