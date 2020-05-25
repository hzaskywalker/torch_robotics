import torch
from robot import tr
from robot.torch_robotics.geometry import simplex

def test_edge_edge():
    s1 = tr.togpu([[0, 0, 0], [0, 0, 0]])
    t1 = tr.togpu([[0, 0, 1], [0, 0, 1]])

    s2 = tr.togpu([[0, 0, 0], [0, 0, 0]])
    t2 = tr.togpu([[0, 1, 0], [0, 0, 1]])
    out = simplex.collide_edge_edge(s1,t1,s2,t2, eps=-1e8)
    print(out)

    mid = torch.randn((1000, 3)) * 3
    vec1 = (torch.rand((1000, 3)) - 0.5)*3
    vec2 = (torch.rand((1000, 3)) - 0.5)*3

    s1 = mid - vec1 * torch.rand((1000, 1))
    t1 = mid + vec1 * torch.rand((1000, 1))

    s2 = mid - vec2 * torch.rand((1000, 1))
    t2 = mid + vec2 * torch.rand((1000, 1))

    dist, pose = simplex.collide_edge_edge(s1, t1, s2, t2)
    inter = pose[..., :3, 3]
    assert dist.abs().max() < 2e-3, f"{dist.abs().max()}"
    assert (inter - mid).abs().max() < 1e-3, f"{(inter-mid).abs().max()}"

def test_face_vertics():

    face = tr.togpu([
        [[1, 0, 0], [1,1,0], [0, 1, 0], [0, 0, 0]]
    ])
    vertices = tr.togpu([
        [0, 0, 1],
        [0, 0, -1],
        [0, 0, 0],
        [0.5, 0.5, 0],
        [1., 1., 0],
        [1., 1., 1.2],
        [1., 1.1, 1.2],
        [3., 3., 1.2],
    ])
    face = face.expand(vertices.shape[0], -1, -1)
    print(simplex.collide_face_vertex(face, vertices, eps=-1e-8))

def test_box():
    #print(simplex.collide_edge_edge(tr.togpu([[0.5, 0.5, 0]]), tr.togpu([[-0.5, 0.5, 0]]),
    #                                                                      tr.togpu([[0.4, 0.1, 0]]), tr.togpu([[0.4, 0.9, 0]])))

    box = simplex.Box(tr.togpu([[[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]]]), size=tr.togpu([[1, 1, 1]]))

    #h = 0.99
    h = 1.1
    box2 = simplex.Box(tr.togpu([[[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0.5+h/2-0.001],
                                 [0, 0, 0, 1]]]), size=tr.togpu([[h, h, h]]))

    print(box.collide_box(box, box2))

def test_box_ground():
    box = simplex.Box(tr.togpu([[[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0.6],
                                 [0, 0, 0, 1]]]), size=tr.togpu([[1, 1, 1]]))

    box.index = torch.arange(1)
    out = simplex.SimpleCollisionDetector().collide_box_ground(box, None)
    print(out)

def test_box_point():
    from robot.torch_robotics.geometry.simplex import SimpleCollisionDetector
    geo = SimpleCollisionDetector()

    rand_pos = tr.Rp_to_trans(tr.projectSO3(torch.randn((6, 3, 3), device='cuda:0')).double(),
                              torch.randn((6, 3), device='cuda:0').double())
    rand_pos = tr.eyes_like(rand_pos)
    box = geo.box(tr.togpu([[[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0.1],
                                 [0, 0, 0, 1]]]).expand(6, -1, -1),
                  size=tr.togpu([[1.2, 0.8, 1.3]]).expand(6, -1))
    sphere = geo.sphere(tr.togpu([[0, 0, 0.5],
                                  [0, 0.5, 0],
                                  [0.5, 0, 0],
                                  [0, 0, -0.5],
                                  [0, -0.5, 0],
                                  [-0.5, 0, 0],
                                  ]),
                        tr.togpu([0.001]).expand(6))
    box.pose = tr.dot(rand_pos, box.pose)
    sphere.pose = tr.dot(rand_pos, sphere.pose)
    out = geo.collide_box_point(box, sphere)
    print(out)
    #print(tr.dot(tr.inv_trans(rand_pos[out[0]]), out[-1]))


if __name__ == '__main__':
    #test_edge_edge()
    #test_face_vertics()
    #test_box()
    #test_box_ground()
    test_box_point()