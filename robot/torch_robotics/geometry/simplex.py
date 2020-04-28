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
        return arith.Rp_to_trans(arith.eyes_like(self.center[..., None, :]), self.center)

    @pose.setter
    def pose(self, pose):
        self.center = pose[..., :3,3]


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
