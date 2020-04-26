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

    def apply_local(self, wrench, mode='set'):
        # we need to transform the wrench
        wrench = arith.transform_wrench(wrench, arith.inv_trans(self.cmass))
        self.apply(wrench, mode)

    def apply(self, wrench, mode='set'):
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

    def kinetic(self):
        return self._rigid_body[self.slice].kinetic()


class Engine:
    def __init__(self, vis=False, gravity=arith.togpu([0, 0, -9.8]),
                 dt = 0.01, frameskip=1,
                 integrator='Euler'):
        assert integrator in ['Euler']
        self.geometry = SimpleCollisionDetector()
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


    @property
    def wrench(self):
        if self._wrench is None:
            self._wrench = self._rigid_bodies.cmass.new_zeros((len(self._rigid_bodies), 6))
        return self._wrench

    def dynamcis(self):
        # Note we don't organize it into batch, we will do it later..
        assert self.n_articulation == 0
        invM, c = self._rigid_bodies.dynamics(gravity=self.gravity)
        return invM, c

    def forward_kinematics(self, shape=True, render=True):
        # The goal is to compute the pose to update the shape and the render
        # However for
        for name in self.objects:
            pose = self.objects[name].fk()
            if shape:
                self.shapes[name].set_pose(pose)
            if render and pose is not None:
                self.visuals[name].set_pose(arith.tocpu(pose[0]))


    def collide(self):
        # call this to make sure that all the rigid bodis are concatenated
        # we should sync

        values = list(self.shapes.values())
        ds = []
        poses = []
        edges = []
        for i in range(len(values)):
            for j in range(i+1, len(values)):
                d, pose, edge = self.geometry.collide(values[i], values[j])

                ds.append(d)
                poses.append(pose)
                edges.append(edge)

        return torch.cat(ds), torch.cat(poses), torch.cat(edges)

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
        shape.index = torch.arange(_slice.start, _slice.stop)
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
        visual = self.renderer.sphere(
            arith.tocpu(center[0]), arith.tocpu(radius[0]), color, name)
        return self.add_rigid_body(cmass, _inertia, mass, shape, visual, name=name)

    def ground(self):
        ground = self.geometry.ground()
        visual = self.renderer.box((10, 10, 1), (255, 255, 255), self.renderer.translate((0, 0, -0.5)))
        return self._add_object(Imortal(), ground, visual, 'ground')

    def render(self, mode='human'):
        self.renderer.render(mode=mode)

    def qacc(self):
        # compute the qacc for rigid bodies and the articulation (not implemented yet)
        self.forward_kinematics()
        invM, c = self.dynamcis()
        qacc = arith.dot(invM, c)

        # TODO: if there are articulation, we should also return the articulation's qacc
        return qacc

    def step(self):
        # perhaps we have to apply force
        for i in range(self.frameskip):
            qacc = self.qacc()
            self._rigid_bodies.euler(qacc, self.dt)
