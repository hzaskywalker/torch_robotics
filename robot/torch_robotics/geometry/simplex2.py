# this the most simple collision checker, we support the collision between the following objects
#   - mesh: triangles or squares that with normal
#   - sphere:
import torch
import logging
from .. import arith


class RigidBody:
    def __init__(self):
        self._index = None

    def set_pose(self, pose):
        self.pose = pose

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index):
        self._index = index

    @property
    def pose(self):
        raise NotImplementedError

    @pose.setter
    def pose(self, pose):
        raise NotImplementedError


class Sphere(RigidBody):
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        super(Sphere, self).__init__()

    @property
    def pose(self):
        return arith.translate(self.center)

    @pose.setter
    def pose(self, pose):
        self.center = pose[..., :3, 3]


class Box(RigidBody):
    # very complex
    VERTICES = [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
                [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1]]
    FACES = [[0, 1, 2, 3], [7, 6, 5, 4], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]]
    EDGES = [[0, 1], [1, 2], [2, 3], [3, 0],
             [4, 5], [5, 6], [6, 7], [7, 4],
             [0, 4], [1, 5], [2, 6], [3, 7], ]
    EDGE_FACE = [[0, 2], [0, 3], [0, 4], [0, 5], [1, 2], [1, 3], [1, 4], [1, 5], [2, 5], [2, 3], [3, 4], [4, 5]]

    def __init__(self, pose, size):
        # 0-1-2-3-0 4-5-6-7-4 0-4 1-5 2-6 3-7
        self.size = size
        self.edges = []
        self.vertices = torch.tensor(self.VERTICES, dtype=pose.dtype, device=pose.device) / 2
        self.faces = torch.tensor(self.FACES, dtype=torch.long, device=pose.device)
        self.edges = torch.tensor(self.EDGES, dtype=torch.long, device=pose.device)
        self.edge_faces = torch.tensor(self.EDGE_FACE, dtype=torch.long, device=pose.device)
        self.pose = pose
        super(Box, self).__init__()

    def get_all(self):
        # return the set of vertices, eges, and poses
        # vertices = self.pose @ (self.vertices[None, :] * self.size)
        # print(vertices)
        # vertices (b, 8, 3)
        # edges (b, 12, 2, 3)
        # faces (b, 6, 4, 3)
        vertices = self.vertices[None, :] * self.size
        vertices = self.pose[:, None, :3, :3].expand(-1, 8, -1, -1) @ vertices[..., None]
        vertices = vertices[..., 0] + self.pose[:, None, :3, 3]

        edges = vertices[:, self.edges]

        faces = vertices[:, self.faces]
        # normal = torch.cross(faces[..., 1, :] - faces[..., 0, :], faces[..., 2, :] - faces[..., 0, :])
        return vertices, edges, faces

    @property
    def pose(self):
        return self._pose

    @pose.setter
    def pose(self, pose):
        self._pose = pose


class Ground(RigidBody):
    # it's always at z coordinates ..
    @property
    def pose(self):
        return None

    @pose.setter
    def pose(self, p):
        assert p is None, "You can't set the pose of the ground!!!"


class SimpleCollisionDetector2:
    def __init__(self, epsilon=1e-5):
        self.epsilon = epsilon
        self.shapes = []

    def register(self, shape):
        self.shapes.append(shape)

    def sphere(self, center, radius):
        return Sphere(center, radius)

    def box(self, pose, size):
        return Box(pose, size)

    def ground(self):
        return Ground()

    def filter(self):
        pass

    def __call__(self, a, b):
        if isinstance(a, Ground):
            a, b = b, a

        if isinstance(a, Sphere) and isinstance(b, Sphere):
            out = self.collide_sphere_sphere(a, b)
        elif isinstance(a, Sphere) and isinstance(b, Ground):
            out = self.collide_sphere_ground(a, b)
        elif isinstance(a, Box) and isinstance(b, Ground):
            out = self.collide_box_ground(a, b)
        else:
            raise NotImplementedError

        # -1 for no object
        batch_id, dist, pose = out
        idx = (dist < self.epsilon)
        return out[idx], dist[idx], pose[idx]

    def collide_sphere_sphere(self, a: Sphere, b: Sphere):
        d = ((a.center - b.center) ** 2 + 1e-16).sum(dim=-1) ** 0.5
        assert (d < 1e-10).sum() == 0, "we don't allow two ball coincident with each other ..."
        vec = (b.center - a.center) / d[:, None]

        p1 = a.center + a.radius[:, None] * vec
        p2 = b.center - b.radius[:, None] * vec

        p = (p1 + p2) / 2  # the contact point...
        pose = arith.Rp_to_trans(arith.normal2pose(-vec), p)
        return torch.arange(d.shape[0], device=d.device), d - a.radius - b.radius, pose

    def collide_sphere_ground(self, a: Sphere, b: Ground):
        # normal should point to the direction that increase the distance
        # pose.dot([1, 0, 0, 1]) is the nomal
        r = a.radius
        d = a.center[..., 2] - r
        pose = a.center.new_zeros((*d.shape, 4, 4))
        pose[..., 0, 2] = -1
        pose[..., 1, 1] = 1
        pose[..., 2, 0] = 1
        pose[..., 3, 3] = 1
        pose[..., :2, 3] = a.center[..., :2]
        return torch.arange(d.shape[0], device=d.device), d, pose

    def collide_box_ground(self, box: Box, b: Ground):
        # we only allow at most 4 vertices under the ground
        # h is the tolerance
        raise NotImplementedError("batch idx for box ground collision is not ")

        vertex, _, _ = box.get_all()
        d = vertex[..., -1]
        is_collision = d < 0 + self.epsilon

        idx = torch.where(is_collision)

        obj_idx = box.index[idx[0]]
        edge = torch.stack((obj_idx, obj_idx * 0 - 1), dim=1)

        dist = d[idx[0], idx[1]]

        pose = vertex.new_zeros((idx[0].shape[0], 4, 4))
        pose[..., 0, 2] = -1
        pose[..., 1, 1] = 1
        pose[..., 2, 0] = 1
        pose[..., 3, 3] = 1
        pose[..., :2, 3] = vertex[idx[0], idx[1], :2]

        #torch.arange(d.shape[0], device=d.device), dist, pose
