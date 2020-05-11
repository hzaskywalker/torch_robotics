import torch
from collections import OrderedDict
from . import arith as tr
from .dynamics import Collision, Mechanism
from .geometry import SimpleCollisionDetector
from ..renderer import Renderer
from .objects import RigidBody, Imortal, Articulation


class ObjectBuilder:
    def __init__(self, epsilon):
        self.geometry = SimpleCollisionDetector(epsilon)
        self.renderer = Renderer()

    # --------------------------------- various geometry -----------------------------------------
    def sphere(self, center, inertia, mass, radius, color, name=None):
        # short cut for creating the sphere objects ...
        # inertia should be the diagonal ...
        assert len(center.shape) == 2
        assert len(inertia.shape) == 2
        assert len(radius.shape) == 1
        assert len(mass.shape) == 1

        cmass = tr.translate(center)
        _inertia = cmass.new_zeros((center.shape[0], 3, 3))
        _inertia[..., [0, 1, 2], [0, 1, 2]] = inertia
        shape = self.geometry.sphere(center, radius)

        visual = self.renderer.compose(self.renderer.sphere(
            tr.tocpu(center[0] * 0), tr.tocpu(radius[0]), color, name + '_sphere'),
            self.renderer.axis(self.renderer.identity(), scale=tr.tocpu(radius[0]) * 1.2),
            name=name)
        visual.set_pose(tr.tocpu(cmass[0]))

        return RigidBody(cmass, _inertia, mass), shape, visual

    def box(self, pose, inertia, mass, size, color, name=None):
        cmass = pose
        _inertia = cmass.new_zeros((cmass.shape[0], 3, 3))
        _inertia[..., [0, 1, 2], [0, 1, 2]] = inertia
        shape = self.geometry.box(cmass, size)

        visual = self.renderer.compose(self.renderer.box(
            tr.tocpu(size[0]), color, self.renderer.identity(), name + '_box'),
            # self.renderer.axis(self.renderer.identity(), scale=tr.tocpu(size[0].max())*1.2),
            name=name)
        visual.set_pose(tr.tocpu(cmass[0]))

        return RigidBody(cmass, _inertia, mass), shape, visual

    def ground(self, ground_size=10):
        ground = self.geometry.ground()
        visual = self.renderer.box((ground_size, ground_size, 1), (255, 255, 255),
                                   self.renderer.translate((0, 0, -0.5)))
        return Imortal(), ground, visual


class Engine2:
    # We write it into a functional
    # Engine2 is a module, we run reset to reset the objects in the world
    def __init__(self, collision_detector, contact_model,
                 renderer = None,
                 contact_dof = 1,
                 gravity=tr.togpu([0, 0, -9.8]),
                 dt=0.01, frameskip=1):

        self.rigid_body = None
        self.objects = self.shapes = self.visuals = None

        self.contact_dof = contact_dof
        self.gravity = gravity
        self.dt = dt
        self.frameskip = frameskip

        self.collision_detector = collision_detector
        self.contact_solver = contact_model
        self.renderer = renderer

    def reset(self, objects, shapes=None, visuals=None):
        # objects is a list
        # shapes is a list
        # visuals is also a list
        self.objects = objects
        if visuals is not None:
            # update visuals
            self.visuals = visuals
        if shapes is not None:
            self.shapes = shapes

        rigid_bodies = []
        collision_shape = []
        articulation_shape = None
        for val, shape in zip(self.objects, self.shapes):
            if isinstance(val, Articulation):
                self.articulation = val
                articulation_shape = shape.sub_shapes()
                continue
            if isinstance(val, RigidBody):
                shape.index = len(rigid_bodies)
                rigid_bodies.append(val)
            else:
                shape.index = -1
            collision_shape.append(shape)
        if articulation_shape is not None:
            idx = len(rigid_bodies)
            for i in articulation_shape:
                i.index = idx; idx += 1
                collision_shape.append(i)
        self.rigid_body = rigid_bodies[0].stack(rigid_bodies, 1)
        self.collision = Collision(collision_shape, self.collision_detector)
        self.mechanism = Mechanism(self.rigid_body, self.articulation, self.contact_dof)
        return self

    def forward_kinematics(self, render_id=None):
        # The goal is to compute the pose to update the shape and the render
        # However for
        for obj, shape, visual in zip(self.objects, self.shapes, self.visuals):
            pose = obj.fk()
            if shape is not None:
                shape.set_pose(pose)
            if render_id is not None and pose is not None and visual is not None:
                visual.set_pose(tr.tocpu(pose[render_id]))

    def render(self, mode='human', render_idx=0):
        self.forward_kinematics(render_id=render_idx)
        return self.renderer.render(mode=mode)

    def do_simulation(self):
        # compute the qacc for rigid bodies and the articulation (not implemented yet)
        self.forward_kinematics() # update collision shapes
        self.collision.update()

        # we may need to add toi
        qacc_obj, qacc_art = self.mechanism(self.gravity, self.collision, tau=None, wrench=None).solve(self.dt)
        self.rigid_body.euler_(qacc_obj, self.dt)
        self.articulation.euler_(qacc_art, self.dt)

    def step(self):
        # perhaps we have to apply force
        for i in range(self.frameskip):
            self.do_simulation()


