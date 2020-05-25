# the collision detector based on the gradient, we need some methods to compute the gradient of the contact point...
# we use a padding schema here ...
import sys
import torch
import numpy as np
from .. import arith as tr

class RigidBody:
    # only rigid body can move
    def __init__(self, c_ptr):
        self.c_ptr = c_ptr

    def set_pose(self, pose):
        self.pose = pose
        self.c_ptr.set_pose(tr.tocpu(pose))

    def get_pose(self):
        return self.pose

class Ground:
    def __init__(self, c_ptr):
        self.c_ptr = c_ptr

    def set_pose(self, pose):
        pass

    def get_pose(self, pose):
        return None


class Simplex:
    def __init__(self, contact_threshold):
        if contact_threshold > 0:
            import logging
            logging.warning("currently we only support contact threshold == 0")

        sys.path.append('/home/hza/Simplex/src/')
        import simplex_c
        self.sim = simplex_c.Simplex(0)
        self.contact_threshold = contact_threshold
        self._detected_collisions = False

        # result of collision
        self.batch_id = None
        self.contact_id = None
        self.contact_objects = None
        self.dist = None
        self.pose = None
        self.max_nc = None

    def box(self, pose, size):
        rigid_body = RigidBody(self.sim.box(tr.tocpu(size[0]) + self.contact_threshold * 2))
        rigid_body.set_pose(pose)
        return rigid_body

    def sphere(self, center, radius):
        pose = tr.translate(center)
        rigid_body = RigidBody(self.sim.sphere(tr.tocpu(radius[0]) + self.contact_threshold))
        rigid_body.set_pose(pose)
        return rigid_body

    def ground(self, h=0):
        box = self.sim.box(np.array([200, 200, 1 + self.contact_threshold * 2]))
        box.set_pose(np.array([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, h-0.5], [0, 0, 0, 1]]]))
        return Ground(box)

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
        self.max_nc = 0

        if update:
            self.update()

        return self

    def update(self):
        self.sim.collide()
        if len(self.sim.batch) > 0:
            self.batch_id = torch.tensor(self.sim.batch, dtype=torch.long, device=self.device)
            self.contact_id = torch.tensor(self.sim.contact_id, dtype=torch.long, device=self.device)
            self.contact_objects = self.shape_idx[torch.tensor(self.sim.object_pair,
                                                               dtype=torch.long, device=self.device)]
            normal_pos = torch.tensor(self.sim.normal_pos, dtype=self.dtype, device=self.device)
            self.max_nc = self.contact_id.max() + 1

            # NOTE: inverse
            self.dist = torch.tensor(-self.sim.dist, dtype=self.dtype, device=self.device) + self.contact_threshold * 2
            self.pose = tr.Rp_to_trans(tr.normal2pose(-normal_pos[:, 0]), normal_pos[:, 1])
        else:
            self.batch_id = self.contact_id = self.contact_objects = self.dist = self.pose = None
            self.max_nc = 0

        return self


    def filter(self, fn=None):
        if fn is None:
            fn = lambda x: ~torch.isnan(x)

        assert self.max_nc > 0

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
