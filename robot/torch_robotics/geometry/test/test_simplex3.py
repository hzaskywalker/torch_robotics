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


def test_gradient():
    sim = Simplex(0, compute_gradients=True)
    center = tr.togpu([[1, 1, 0, 0],
                       [0, 1, 0, 0],
                       [1, 0, 1, 0],
                       [0, 0, 0, 1]])[None, :]
    size = tr.togpu([1, 1, 1])[None, :]


    center.requires_grad = True
    box = sim.box(center, size)

    center2 = tr.togpu([0, 0, 0.8])[None,:]
    center2.requires_grad = True
    sphere = sim.sphere(center2, tr.togpu([0.5]))
    center2 = tr.togpu([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0.9],
                       [0, 0, 0, 1]])[None, :]
    box2 = sim.box(center2, size*0.99)

    shapes = [box, box2]
    sim.collide(shapes, update=True)

    (sim.dist.sum()+sim.pose.sum()).backward()
    print(center.grad)
    #jac = sim.sim.jacobian.reshape(4, 2, 2, 6, 7)
    #jac = jac.mean(axis=2)

    h = 0.01
    for i in range(4):
        for j in range(4):
            center3 = center.clone()
            center3[:, i, j] += h
            box.set_pose(center3)
            sim.update()
            a = sim.dist.sum() + sim.pose.sum()
            center3[:, i, j] -= 2*h
            box.set_pose(center3)
            sim.update()
            b = sim.dist.sum() + sim.pose.sum()
            print(float((a-b)/2/h), end=' ')
        print('')


if __name__ == '__main__':
    test_gradient()