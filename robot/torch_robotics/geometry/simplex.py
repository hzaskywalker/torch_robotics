# this the most simple collision checker, we support the collision between the following objects
#   - mesh: triangles or squares that with normal
#   - sphere:
import torch
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
        self.center = pose[..., :3,3]

def collide_edge_edge(s1, t1, s2, t2, eps=-1e-14, inf=1e9):
    """
    :param s1, t1, s2, t2: (batch, 3)
    :return: dist
    """
    # we solve it with least square
    # \Vert s1 + (t1-s1) * a - (s2 +(t2-s2) * b)\Vert_2^2
    # \min || [t1-s1  -(t2-s2)][a, b]^T - (s2-s1) ||
    assert s1.shape == t1.shape and s1.shape == s2.shape and len(s1.shape) == 2
    vec1, vec2 = t1-s1, t2-s2

    A = torch.stack((vec1, -vec2), dim=-1)
    # solve \min (Ax - b)^T => (A^TA)^{-1}Ab
    AT = A.transpose(-1, -2)

    X = AT@A
    det = X[:, 0, 0] * X[:, 1, 1] - X[:, 1, 0] * X[:, 0, 1]

    colinear = (det.abs() < 1e-12)
    #invX = X.transpose(-1, -2)/(det + colinear)[:, None, None]
    invX = torch.zeros_like(X)
    factor = (det + colinear)[:, None]
    invX[:, [0, 1], [0, 1]] = X[:, [1, 0], [1, 0]]/factor
    invX[:, [0, 1], [1, 0]] = -X[:, [1, 0], [0, 1]]/factor

    # if this is too slow, use A^T/A.diag
    ans = invX@(AT@(s2 - s1)[..., None])
    p1 = s1 + vec1 * ans[:, 0]
    p2 = s2 + vec2 * ans[:, 1]
    normal = torch.cross(vec1, vec2)
    ans = ans[..., 0]
    flag = (ans[:, 0] >= eps) & (ans[:, 0] <= 1-eps) & (ans[:, 1]>=eps) & (ans[:, 1]<=1-eps) & (~colinear)
    R = torch.stack((normal, vec1, vec2), dim=-1)
    if flag.any():
        R[flag] = arith.projectSO3(R[flag])

    dist = ((p1 - p2) ** 2).sum(dim=-1).clamp(0, inf) ** 0.5
    return dist + inf * (~flag).float(), arith.Rp_to_trans(R, (p1+p2)/2)


def collide_face_vertex(faces, vertices, eps=-1e14):
    """
    :param faces: (b, n, points)
    :param vertices: (b, 3)
    :param eps: control the boundary, if eps < 0 means that we allow the point to be slightly beyond the face
    :return: the project along the face normal and if we project the vertices on the face, if it's inside the faces..
    """
    # put them on the origin
    vertices = vertices - faces[:, 0]
    faces = faces - faces[:, 0:1]
    normal = torch.cross(faces[:, 1], faces[:, 2]) # we make sure that the normal is point to the outside ....
    dist = (vertices * normal).sum(dim=-1)/normal.norm(dim=-1) # dist
    vertices -= dist[..., None] * normal # project to the face domain

    A = torch.cat((faces[:, 1:], faces[:, 0:1]), dim=1) - faces
    B = vertices[:, None, :] - faces
    flag = (torch.cross(A, B, dim=-1)*normal[:, None]).sum(dim=-1)
    return dist, (flag >= eps).all(dim=1)


class Box(RigidBody):
    # very complex
    def __init__(self, pose, size):
        # 0-1-2-3-0 4-5-6-7-4 0-4 1-5 2-6 3-7
        self.size = size
        self.edges = []
        self.vertices = torch.tensor(
            [[-1, -1, 1], [1, -1, 1], [1,1,1], [-1, 1, 1],
             [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1]],
            dtype=pose.dtype, device=pose.device
        )/2
        self.faces = torch.tensor(
            [[0, 1, 2, 3], [7, 6, 5, 4], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]],
            dtype= torch.long, device=pose.device
        )
        self.edges = torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 0],
             [4, 5], [5, 6], [6, 7], [7, 4],
             [0, 4], [1, 5], [2, 6], [3, 7],],
            dtype=torch.long, device=pose.device
        )
        self.edge_faces = []
        raise NotImplementedError("Edge faces is not implemented...")
        self.pose = pose
        super(Box, self).__init__()

    def get_all(self):
        # return the set of vertices, eges, and poses
        #vertices = self.pose @ (self.vertices[None, :] * self.size)
        #print(vertices)
        # vertices (b, 8, 3)
        # edges (b, 12, 2, 3)
        # faces (b, 6, 4, 3)
        vertices = self.vertices[None, :] * self.size
        vertices = self.pose[:, None, :3, :3].expand(-1, 8, -1, -1) @ vertices[..., None]
        vertices = vertices[..., 0] + self.pose[:, None,:3,3]

        edges = vertices[:, self.edges]

        faces = vertices[:, self.faces]
        normal = torch.cross(faces[..., 1, :] - faces[..., 0, :], faces[..., 2, :] - faces[..., 0, :])
        print(vertices.shape, edges.shape, faces.shape)
        return vertices, edges, faces

    @classmethod
    def collide_box(cls, box1, box2, edge_tolerance=1e-3):
        # collision between two boxes
        vertices1, edges1, faces1 = box1.get_all()
        vertices2, edges2, faces2 = box2.get_all()

        # first we need to do the edges1 to edges2 check
        shape_old = (edges1.shape[0], edges1.shape[1], edges2.shape[1])
        shape_new = (edges1.shape[0] * edges1.shape[1] * edges2.shape[1], 2, 3)
        E1 = edges1[:, :, None].expand(*shape_old, -1, -1).reshape(shape_new)
        E2 = edges2[:, None, :].expand(*shape_old, -1, -1).reshape(shape_new)
        d, pose = collide_edge_edge(E1[..., 0, :], E1[..., 1, :], E2[..., 0, :], E2[..., 1, :])
        d = d.reshape(*shape_old)
        pose = pose.reshape(*shape_old, 4, 4)
        p = torch.where(d.abs() < edge_tolerance)

        edge_d = d[p[0], p[1], p[2]]
        edge_pose = pose[p[0], p[1], p[2]]

        #TODO: check the face normal to decide the orientation,
        # i.e., we compare the computed normal with the face normal associated with the edge and reverse
        # its direction if we found it's not correct...

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


class SimpleCollisionDetector:
    def __init__(self, epsilon=1e-5):
        self.epsilon = epsilon
        self.shapes = []

    def register(self, shape):
        self.shapes.append(shape)

    def sphere(self, center, radius):
        return Sphere(center, radius)

    def ground(self):
        return Ground()

    def collide(self, a, b):
        if isinstance(a, Ground):
            a, b = b, a

        if isinstance(a, Sphere) and isinstance(b, Sphere):
            dist, pose = self.collide_sphere_sphere(a, b)
        elif isinstance(a, Sphere) and isinstance(b, Ground):
            dist, pose = self.collide_sphere_ground(a, b)
        else:
            raise NotImplementedError

        # -1 for no object
        b_index = (b.index if b.index is not None else a.index*0-1)
        idx = (dist < self.epsilon)

        edge = torch.stack((a.index[idx], b_index[idx]), dim=1)
        return dist[idx], pose[idx], edge


    def collide_sphere_sphere(self, a:Sphere, b: Sphere):
        d = ((a.center-b.center) **2 + 1e-16).sum(dim=-1)**0.5
        assert (d < 1e-10).sum() == 0, "we don't allow two ball coincident with each other ..."
        vec = (b.center - a.center)/d

        p1 = a.center + a.radius * vec
        p2 = b.center - b.radius * vec

        p = (p1 + p2)/2 # the contact point...
        pose = arith.Rp_to_trans(arith.normal2pose(-vec), p)
        return d - a.radius - b.radius, pose

    def collide_sphere_ground(self, a:Sphere, b: Ground):
        # normal should point to the direction that increase the distance
        # pose.dot([1, 0, 0, 1]) is the nomal
        r = a.radius
        d = a.center[..., 2] - r
        pose = a.center.new_zeros((*d.shape, 4, 4))
        pose[..., 0,2] = -1
        pose[..., 1,1] = 1
        pose[..., 2,0] = 1
        pose[..., 3,3] = 1
        pose[..., :2, 3] = a.center[..., :2]
        return d, pose
