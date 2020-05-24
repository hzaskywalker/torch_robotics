# the collision detector based on the gradient, we need some methods to compute the gradient of the contact point...
import sys
import torch
from .. import arith as tr

class RigidBody:
    def __init__(self, c_ptr):
        self.c_ptr = c_ptr
        self.immortal = 0

    def set_pose(self, pose):
        self.pose = pose
        self.c_ptr.set_pose(tr.tocpu(pose))

    def get_pose(self):
        return self.pose


class Simplex:
    def __init__(self, contact_threshold):
        sys.path.append('/home/hza/Simplex/src/')
        import simplex
        self.sim = simplex.Simplex(contact_threshold)
        self._detected_collisions = False

        # result of collision
        self.batch_id = None
        self.contact_id = None
        self.contact_objects = None
        self.dist = None
        self.pose = None
        self.max_nc = None

    def box(self, pose, size):
        rigid_body = RigidBody(self.sim.box(tr.tocpu(size[0])))
        rigid_body.set_pose(pose)
        return rigid_body

    def sphere(self, center, radius):
        pose = tr.translate(center)
        rigid_body = RigidBody(self.sim.Sphere(tr.tocpu(radius[0])))
        rigid_body.set_pose(pose)
        return rigid_body

    def add_shape(self, shape:RigidBody):
        self.sim.add_shape(shape.c_ptr)
        return self

    def clear_shapes(self):
        self.sim.clear_shapes()

    def collide(self, shapes, shape_idx=None, update=True):
        if shape_idx is None:
            shape_idx = list(range(len(shapes)))

        self.clear_shapes()
        for i in shapes:
            self.add_shape(i)

        self.shapes = shapes

        obj = shapes[0].pose
        self.device = obj.device
        self.dtype = obj.dtype
        self.shape_idx = torch.tensor(shape_idx, dtype=torch.long, device=self.device)
        self.max_nc = None

        if update:
            self.update()

        return self

    def update(self):
        self.sim.collide()
        self.batch_id = torch.tensor(self.sim.batch, dtype=torch.long, device=self.device)
        self.contact_id = torch.tensor(self.sim.contact_id, dtype=torch.long, device=self.device)
        self.contact_objects = self.shape_idx[torch.tensor(self.sim.object_pair,
                                                           dtype=torch.long, device=self.device)]
        self.dist = torch.tensor(self.sim.dist, dtype=self.dtype, device=self.device)
        normal_pos = torch.tensor(self.sim.normal_pos, dtype=self.dtype, device=self.device)
        self.pose = tr.Rp_to_trans(tr.normal2pose(normal_pos[:, 0]), normal_pos[:, 1])
        return self


    def filter(self, fn=None):
        if fn is None:
            fn = lambda x: ~torch.isnan(x)

        if self.max_nc is None:
            self.update()

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

        return torch.cat(batch_id), \
               torch.cat(contact_id), \
               torch.cat(pose), \
               torch.cat(obj_id), \
               torch.cat(sign)
