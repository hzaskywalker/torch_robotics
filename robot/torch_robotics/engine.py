# base class that serves as the
import torch
from collections import OrderedDict
from .geometry import SimpleCollisionDetector
from ..renderer import Renderer
from .objects import RigidBody, Imortal
from . import arith


class Handler:
    def __init__(self, engine, name, obj, shape, visual):
        self.name = name
        self.engine = engine
        self.obj = obj
        self.shape = shape
        self.visual = visual


class RigidBodyHandler:
    def __init__(self, slice, rigid_body, engine):
        self.slice = slice
        self._rigid_body = rigid_body
        self.engine = engine

    def apply(self, wrench, mode='set'):
        """
        Notice we use this only for debug as usually we can't manually set a force to the object easily ...
        Once the object starts to rotate, it's hard to decide it's rotation angle ..
        So in general, we will only apply a force to the center of the mass without any torque
        :param wrench: local wrench apply to the rigid's cmass frame..
        """
        if mode == 'set':
            self.engine.wrench[self.slice] = wrench
        else:
            self.engine.wrench[self.slice] += wrench

    def clear(self):
        self.engine.wrench[self.slice] = 0

    @property
    def cmass(self):
        return self._rigid_body.cmass[self.slice]

    @property
    def inertia(self):
        return self._rigid_body.inertia[self.slice]

    @property
    def mass(self):
        return self._rigid_body.mass[self.slice]

    @property
    def velocity(self):
        return self._rigid_body.velocity[self.slice]

    @cmass.setter
    def cmass(self, value):
        self._rigid_body.cmass[self.slice] = value

    @inertia.setter
    def inertia(self, value):
        self._rigid_body.inertia[self.slice] = value

    @mass.setter
    def mass(self, value):
        self._rigid_body.mass[self.slice] = value

    @velocity.setter
    def velocity(self, value):
        self._rigid_body.velocity[self.slice] = value

    def get_qpos(self):
        q = self.cmass[..., :3, [0, 1, 3]]
        return q.reshape(*q.shape[:-2], 9)

    def set_qpos(self, pose):
        cmass = pose.new_zeros((*pose.shape[:-1], 4, 4))
        pose = pose.reshape(*pose.shape[:-1], 3, 3)
        cmass[..., :3, 3] = pose[..., :3, 2]
        cmass[..., :3, :3] = arith.projectSO3(pose)
        cmass[..., 3, 3] = 1
        self.cmass = cmass

    def get_qvel(self):
        return self.velocity

    def set_qvel(self, velocity):
        self.velocity = velocity

    def fk(self):
        return self.cmass

    def __getattr__(self, item):
        return self._rigid_body[self.slice].__getattribute__(item)


class Engine:
    def __init__(self,
                 gravity=arith.togpu([0, 0, -9.8]),
                 dt = 0.01, frameskip=1,
                 integrator='Euler',
                 contact_model=None, epsilon=1e-3
             ):
        assert integrator in ['Euler']
        self.geometry = SimpleCollisionDetector(epsilon)
        self.renderer = Renderer()

        self.objects = OrderedDict() # we should apply force to objs
        self.shapes = OrderedDict()
        self.visuals = OrderedDict()
        self.handlers = OrderedDict()

        self.n_articulation = 0
        self.n_rigid_body = 0

        self._rigid_bodies = None
        self._wrench = None

        self.gravity = gravity
        self.dt = dt
        self.frameskip = frameskip
        self.integrator = integrator

        self.contact_model = contact_model

    @property
    def wrench(self):
        if self._wrench is None:
            self._wrench = self._rigid_bodies.cmass.new_zeros((len(self._rigid_bodies), 6))
        elif self._wrench.shape[0] != len(self._rigid_bodies):
            n = len(self._rigid_bodies) - self._wrench.shape[0]
            self._wrench = torch.cat((self._wrench, self._wrench.new_zeros(n, 6)))
        return self._wrench

    def dynamcis(self):
        # Note we don't organize it into batch, we will do it later..
        assert self.n_articulation == 0
        invM, c = self._rigid_bodies.dynamics(gravity=self.gravity, wrench=self.wrench)
        return invM, c

    def forward_kinematics(self, shape=True, render=None):
        # The goal is to compute the pose to update the shape and the render
        # However for
        for name in self.objects:
            pose = self.objects[name].fk()
            if shape:
                self.shapes[name].set_pose(pose)
            if render is not None and pose is not None:
                self.visuals[name].set_pose(arith.tocpu(pose[render]))


    def collide(self, return_jacobian=True):
        # collision detection
        # if we don't have the contact model or we don't need the jacobian, return the collision detection result
        # otherwise we return the sparse blocked jacobian matrix (jacobian, constraint_id, obj_id)
        values = list(self.shapes.values())
        dists = []
        poses = []
        edges = []
        for i in range(len(values)):
            for j in range(i+1, len(values)):
                d, pose, edge = self.geometry.collide(values[i], values[j])
                if edge.shape[0] > 0:
                    dists.append(d)
                    poses.append(pose)
                    edges.append(edge)

        if len(edges) == 0:
            return None, None
        dists, poses, edges = torch.cat(dists), torch.cat(poses), torch.cat(edges)
        #print('NUM_COLLISION:', len(dists))
        if self.contact_model is None or not return_jacobian:
            return dists, poses, edges
        return dists, self.compute_jacobian(dists, poses, edges)

    def rigid_body_index2xy(self, index):
        total = len(self._rigid_bodies)
        assert total % self.n_rigid_body == 0
        batch_size = total//self.n_rigid_body
        self.batch_size = batch_size
        return index % batch_size, index//batch_size

    def compute_jacobian(self, dist, poses, edges):
        # the jacobian should be organized in the following form:
        # we return the jacobian in form of (6x6xb) and the corresponding constraints id

        # given the poses

        # from torch_geometric.utils import scatter_
        left_index = edges[:, 0]
        right_index = edges[:, 1]

        left_mask = left_index>=0
        assert left_mask.all()

        # we remove imortal objects ...
        right_mask = (right_index >= 0)
        right_index = right_index[right_mask]

        all_index = torch.cat((left_index, right_index), dim=0)
        all_poses = torch.cat([poses, poses[right_mask]], dim=0)

        jac = self._rigid_bodies[all_index].compute_jacobian(all_poses)
        jac[len(left_mask):] *= -1
        #TODO: in fact we can speed it up, but we don't need to do it now...

        constrain_id = torch.cat(torch.where(left_mask>=0) + torch.where(right_mask), dim=0)
        return (jac, constrain_id, all_index)


    def _add_object(self, obj, shape, visual, name=None):
        if name is None:
            name = f"__obj__{len(self.objects)+1}"
        assert name not in self.objects

        if isinstance(obj, RigidBodyHandler):
            self.n_rigid_body += 1
        elif isinstance(obj, Imortal):
            pass
        else:
            raise NotImplementedError

        self.objects[name] = obj
        self.shapes[name] = shape
        self.visuals[name] = visual

        # return the name of the objects...
        handler = Handler(self, name, obj, shape, visual)
        self.handlers[name] = handler
        return handler

    def add_rigid_body(self, cmass, inertia, mass, shape, visual, name=None):
        r = RigidBody(cmass, inertia, mass)

        if self._rigid_bodies is None:
            self._rigid_bodies = r
            start, end = 0, len(self._rigid_bodies)
        else:
            #TODO: I don't know if there is any other better solution
            k = self._rigid_bodies
            start= len(k)
            k.assign(r.cat([k, r]))
            end = len(k)

        _slice = slice(start, end)
        # set the rigid body index for the shape ...
        shape.index = torch.arange(_slice.start, _slice.stop, device=cmass.device)
        return self._add_object(RigidBodyHandler(_slice, self._rigid_bodies, self),
                                shape, visual, name)

    def sphere(self, center, inertia, mass, radius, color, name=None):
        # short cut for creating the sphere objects ...
        # inertia should be the diagonal ...
        assert len(center.shape) == 2
        assert len(inertia.shape) == 2
        assert len(radius.shape) == 1
        assert len(mass.shape) == 1

        cmass = arith.translate(center)
        _inertia = cmass.new_zeros((center.shape[0], 3, 3))
        _inertia[...,[0, 1, 2],[0, 1, 2]] = inertia
        shape = self.geometry.sphere(center, radius)

        visual = self.renderer.compose(self.renderer.sphere(
            arith.tocpu(center[0]*0), arith.tocpu(radius[0]), color, name+'_sphere'),
            self.renderer.axis(self.renderer.identity(), scale=arith.tocpu(radius[0])*1.2),
            name=name)
        visual.set_pose(arith.tocpu(cmass[0]))

        return self.add_rigid_body(cmass, _inertia, mass, shape, visual, name=name)

    def box(self, pose, inertia, mass, size, color, name=None):
        cmass = pose
        _inertia = cmass.new_zeros((cmass.shape[0], 3, 3))
        _inertia[...,[0, 1, 2],[0, 1, 2]] = inertia
        shape = self.geometry.box(cmass, size)

        visual = self.renderer.compose(self.renderer.box(
            arith.tocpu(size[0]), color, self.renderer.identity(), name+'_box'),
            #self.renderer.axis(self.renderer.identity(), scale=arith.tocpu(size[0].max())*1.2),
            name=name)
        visual.set_pose(arith.tocpu(cmass[0]))

        return self.add_rigid_body(cmass, _inertia, mass, shape, visual, name=name)

    def ground(self, ground_size=10):
        ground = self.geometry.ground()
        visual = self.renderer.box((ground_size, ground_size, 1), (255, 255, 255), self.renderer.translate((0, 0, -0.5)))
        return self._add_object(Imortal(), ground, visual, 'ground')

    def render(self, mode='human', render_idx=0):
        self.forward_kinematics(render=render_idx)
        return self.renderer.render(mode=mode)

    def do_simulation(self):
        # compute the qacc for rigid bodies and the articulation (not implemented yet)
        self.forward_kinematics()
        invM, c = self.dynamcis()
        # TODO: if there are articulation, we should also return the articulation's qacc
        qacc = arith.dot(invM, c)

        use_toi = False
        if self.contact_model:
            dist, jac = self.collide()
            if jac is not None:
                output = self.contact_model(self, jac, invM, c, dist, self._rigid_bodies.velocity)
                if not isinstance(output, tuple):
                    qacc = qacc + output
                else:
                    use_toi = True
                    qacc_f, toi = output
        if not use_toi:
            self._rigid_bodies.euler(qacc, self.dt)
        else:
            assert qacc.shape == qacc_f.sum(dim=-1).shape
            qacc_f = qacc_f.transpose(-1, -2)
            index = toi.argsort(dim=-1)
            toi = toi.gather(index=index, dim=1)
            qacc_f = qacc_f.gather(index=index[:, :, None].expand_as(qacc_f), dim=1)

            steps = toi.shape[1]
            now = 0
            for i in range(steps+1):
                next = toi[:, i] if i < steps else self.dt
                self._rigid_bodies.euler(qacc, next-now)
                now = next
                if i < steps:
                    qacc = qacc + qacc_f[:, i]

    def step(self):
        # perhaps we have to apply force
        for i in range(self.frameskip):
            self.do_simulation()
