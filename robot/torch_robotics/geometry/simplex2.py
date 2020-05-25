# this the most simple collision checker, we support the collision between the following objects
#   - mesh: triangles or squares that with normal
#   - sphere:
import torch
import numpy as np
from .. import arith


class RigidBody:
    def set_pose(self, pose):
        self.pose = pose

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


class Simplex:
    def __init__(self, epsilon=1e-5):
        self.epsilon = epsilon
        self.shapes = []

    def register(self, shape):
        self.shapes.append(shape)

    def sphere(self, center, radius):
        assert len(radius.shape) == 1, "radius must be a 1-d vector"
        return Sphere(center, radius)

    def box(self, pose, size):
        return Box(pose, size)

    def ground(self):
        return Ground()

    def filter(self):
        pass

    def __call__(self, a, b):
        swap = False
        if isinstance(a, Ground) or (isinstance(a, Sphere) and isinstance(b, Box)):
            a, b = b, a
            swap = True

        if isinstance(a, Sphere) and isinstance(b, Sphere):
            out = self.collide_sphere_sphere(a, b)
        elif isinstance(a, Sphere) and isinstance(b, Ground):
            out = self.collide_sphere_ground(a, b)
        elif isinstance(a, Box) and isinstance(b, Ground):
            out = self.collide_box_ground(a, b)
        elif isinstance(a, Box) and isinstance(b, Sphere):
            out = self.collide_box_point(a, b)
        else:
            raise NotImplementedError
        # -1 for no object
        batch_id, dist, pose = out
        idx = (dist < self.epsilon)
        if idx.any() and len(idx)>0:
            return batch_id[idx], dist[idx], pose[idx], swap
        else:
            return batch_id[idx], None, None, swap

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
        vertex, _, _ = box.get_all()
        d = vertex[..., -1]
        is_collision = d < 0 + self.epsilon

        idx = torch.where(is_collision)

        batch_id = torch.arange(d.shape[0], device=d.device)

        batch_id = batch_id[idx[0]]
        dist = d[idx[0], idx[1]]

        pose = vertex.new_zeros((idx[0].shape[0], 4, 4))
        pose[..., 0, 2] = -1
        pose[..., 1, 1] = 1
        pose[..., 2, 0] = 1
        pose[..., 3, 3] = 1
        pose[..., :2, 3] = vertex[idx[0], idx[1], :2]

        #torch.arange(d.shape[0], device=d.device), dist, pose
        return batch_id, dist, pose

    def collide_box_point(self, box: Box, sphere: Sphere):
        # the contact point is the center of the sphere
        assert (sphere.radius <= 0.2).all(), "We only support the collision between the box and the very small sphere"
        box_size = box.size / 2
        box_pose = box.pose
        box_size = box_size + sphere.radius[:, None] + self.epsilon
        point = arith.dot(arith.inv_trans(box_pose), sphere.pose)[..., :3, 3]

        batch_id = torch.arange(point.shape[0], device=point.device)
        inside = ((point <= box_size) & (point >= -box_size)).all(dim=-1)

        if inside.any():
            diffs = torch.cat((box_size-point, point + box_size), dim=-1)[inside] # (3, 3)
            dist, face_id = diffs.min(dim=-1)
            dist = -(dist-self.epsilon)
            face_id = face_id
            p = point[inside]

            normal = torch.zeros_like(p)
            tmp = batch_id[:p.shape[0]]
            normal[tmp, face_id%3] = -(1 - 2*(face_id//3).double())
            pose = arith.dot(box_pose[inside], arith.Rp_to_trans(arith.normal2pose(normal), p))
        else:
            pose = None
            dist = batch_id[inside].float()
        return batch_id[inside], dist, pose

    def collide(self, shapes, shape_idx=None):
        return Collision(shapes, shape_idx, self)


class Collision:
    def __init__(self, shapes, shape_idx, collision_checker):
        # each shape has an object id
        self.shapes = shapes
        self.shape_idx = shape_idx
        self.collision_checker = collision_checker

        self.max_nc = 0
        # in the form of (num_contacts,)
        self.batch_id = []
        self.contact_id = []
        self.dist = []
        self.pose = []

        # list of object_id and contact_id
        # (num_contact, 2)
        # represent the two object id of contact
        # num_of_objects + num of links in articulation
        self.contact_objects = []

    def update(self):
        shapes = self.shapes
        for i in shapes:
            if i.pose is not None:
                batch_size = i.pose.shape[0]
                break

        # count the number of contacts per batch
        batch_nc = np.zeros((batch_size,), dtype=np.int32) # contact id in batch
        self.batch_id, self.contact_id, self.dist, self.pose, self.contact_objects = [], [], [], [], []

        for i in range(len(self.shapes)):
            a_id = self.shape_idx[i]
            for j in range(i+1, len(shapes)):
                b_id = self.shape_idx[j]

                batch_id, dists, poses, swap = self.collision_checker(shapes[i], shapes[j])

                if batch_id.shape[0] > 0:
                    # number of contacts
                    _contact_id = np.zeros(len(batch_id), dtype=np.int32)
                    for idx, b in enumerate(batch_id.detach().cpu().numpy()):
                        _contact_id[idx] = batch_nc[b]
                        batch_nc[b] += 1
                    # currently a tensor
                    self.batch_id.append(batch_id)
                    self.dist.append(dists)
                    self.pose.append(poses)

                    # current a np array
                    self.contact_id.append(_contact_id)
                    tmp = a_id, b_id
                    if swap:
                        tmp = b_id, a_id
                    self.contact_objects.append(np.array(tmp)[None,:].repeat(len(batch_id), 0))

        self.max_nc = 0
        if len(self.batch_id)>0:
            self.max_nc = batch_nc.max()
            self.batch_id = torch.cat(self.batch_id, dim=0)
            self.dist = torch.cat(self.dist, dim=0)
            self.pose = torch.cat(self.pose, dim=0)
            device = self.batch_id.device
            self.contact_id = torch.tensor(np.concatenate(self.contact_id), dtype=torch.long, device=device)
            self.contact_objects = torch.tensor(np.concatenate(self.contact_objects), dtype=torch.long, device=device)

        return self

    def filter(self, fn):
        object_mask = fn(self.contact_objects)
        batch_id, contact_id, sign, pose, obj_id = [], [], [], [], []
        for i in range(2):
            mask = object_mask[:, i]
            if mask.any():
                batch_id.append(self.batch_id[mask])
                contact_id.append(self.contact_id[mask])
                sign.append(torch.zeros_like(self.dist[mask]) * 0 + (1 - i * 2))
                pose.append(self.pose[mask])
                obj_id.append(self.contact_objects[:, i][mask])
        return torch.cat(batch_id), torch.cat(contact_id), torch.cat(pose), torch.cat(obj_id), torch.cat(sign)
