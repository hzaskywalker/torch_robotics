# visualzation tools for the articulations ...
from robot import tr
import numpy as np
from ..renderer import Renderer
from .objects.articulation import Articulation

class EndEffectorShape:
    def __init__(self, link_id, ee_shape):
        # we only care about the end effector's shape
        self.ee_shape = ee_shape
        self.link_id = link_id

    def sub_shapes(self):
        return [(self.link_id, self.ee_shape)]

    def set_pose(self, pose):
        p = pose[:, self.link_id]
        self.ee_shape.set_pose(p)

class ArmShapes:
    def __init__(self, shapes):
        self.shapes = shapes

    def set_pose(self, pose):
        for a, b in zip(self.shapes, pose):
            a.set_pose(b)

def make_robots(geometry, renderer: Renderer):
    M = tr.togpu([[
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, -0.5],
            [0, 0, 0, 1],
        ],
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, -1.0],
            [0, 0, 0, 1],
        ],
        [[1, 0, 0, 0],
        [0, 1, 0, 0.],
        [0, 0, 1, -0.5],
        [0, 0, 0, 1.],]
    ]])

    G = tr.togpu([
       [np.diag([1, 1, 1, 1, 1, 1]), np.diag([1, 1, 1, 1, 1, 1])]
    ])

    w, q = [1, 0, 0], [0, 0, 0.5]
    screw1 = w + (-np.cross(w, q)).tolist()
    A = tr.togpu([[screw1, screw1]])

    qpos = tr.togpu([[0, 0]])
    qvel = tr.togpu([[0, 0]])


    shapes = []
    for i in range(len(M[0])-1):
        length = abs(A[0, i].detach().cpu().numpy()[4]) * 2
        capsule = renderer.capsule(length, 0.1, (255, 255, 255, 127), renderer.x2z())
        shapes.append(renderer.compose(capsule, renderer.sphere((0, 0, -length/2), 0.1, (255, 0, 0))))

    shapes.append(renderer.sphere(np.array([0, 0, 0]), 0.12, (0, 0, 255)))
    visual = ArmShapes(shapes)

    articulation = Articulation(qpos, qvel, M, A, G)

    ball = geometry.sphere(tr.togpu([[0, 0, 0]]), tr.togpu([0.12]))
    return articulation, EndEffectorShape(len(M[0])-1, ball), visual
