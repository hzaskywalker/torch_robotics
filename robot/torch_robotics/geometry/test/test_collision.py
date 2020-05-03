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
    box = simplex.Box(tr.togpu([[[0, 0, 1, 2],
                         [0, 1, 0, 2],
                         [1, 0, 0, 2],
                         [0, 0, 0, 1]]]), size=tr.togpu([[0.5, 0.6, 0.3]]))
    box.get_all()


if __name__ == '__main__':
    #test_edge_edge()
    #test_face_vertics()
    test_box()