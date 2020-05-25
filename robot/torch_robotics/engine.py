import torch
from collections import OrderedDict
from . import arith as tr
from .dynamics import Mechanism
from ..renderer import Renderer
from .objects import RigidBody, Imortal, Articulation

from .geometry.simplex import Simplex
from robot.torch_robotics.contact.elastic import StewartAndTrinkle


class Builder:
    def __init__(self, epsilon=None):
        self.geometry = Simplex(epsilon)
        self.renderer = Renderer()

        # default renderer
        self.renderer.axis(self.renderer.identity())
        self.renderer.set_camera_position(-10, 0, 0)
        self.renderer.set_camera_rotation(0, 0)


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

    def robot(self, robot_name):
        from .robots import make_robots
        return make_robots(self.geometry, self.renderer)


class Engine(Builder):
    # We write it into a functional
    # Engine2 is a module, we run reset to reset the objects in the world
    ContactModel = StewartAndTrinkle
    def __init__(self, contact_dof=1, contact_model=None,
                 collision_threshold=0.001, contact_margine=0,
                 gravity=tr.togpu([0, 0, -9.8]),
                 dt=0.01, frameskip=1, mu=0, restitution=1):
        if contact_model is None:
            assert contact_margine is not None, "Please either set the contact model or set the contact margine"
            contact_model = self.ContactModel(alpha0=contact_margine, mu=mu, restitution=restitution)
        super(Engine, self).__init__(collision_threshold)

        self.init_objects = []
        # the above only stores the initialization of the objects, we won't use it for forward dynamics

        self.shapes = []
        self.visuals = []
        self.rigid_body: RigidBody = None
        self.articulation: Articulation = None


        self.contact_dof = contact_dof
        self.gravity = gravity
        self.dt = dt
        self.frameskip = frameskip

        self.collision_detector = self.geometry
        self.contact_model = contact_model

    def add(self, obj, shape, visual):
        self.init_objects.append(obj)
        self.shapes.append(shape)
        self.visuals.append(visual)
        return self

    def reset(self, objects=None, shapes=None, visuals=None):
        # objects is a list
        # shapes is a list
        # visuals is also a list
        if objects is not None:
            self.init_objects = objects
        if visuals is not None:
            self.visuals = visuals
        if shapes is not None:
            self.shapes = shapes

        assert len(self.shapes) is not None, "we need at least one objects"
        assert len(self.shapes) == len(self.init_objects), "please provide the collision shape"

        rigid_bodies = []
        collision_shape = []
        collision_shape_index = []
        articulation_shape = None
        for val, shape in zip(self.init_objects, self.shapes):
            if isinstance(val, Articulation):
                self.articulation = val
                articulation_shape = shape.sub_shapes()
                continue
            if isinstance(val, RigidBody):
                index = len(rigid_bodies)
                rigid_bodies.append(val)
            else:
                index = -1
            collision_shape.append(shape)
            collision_shape_index.append(index)
        if articulation_shape is not None:
            idx = len(rigid_bodies)
            for link_id, i in articulation_shape:
                collision_shape_index.append(link_id + idx)
                collision_shape.append(i)
        if len(rigid_bodies) > 0:
            self.rigid_body = rigid_bodies[0].stack(rigid_bodies, 1)
        self.collision = self.geometry.collide(collision_shape, collision_shape_index)
        self.mechanism = Mechanism(self.rigid_body, self.articulation, self.contact_dof,
                                   contact_method=self.contact_model)
        return self

    def forward_kinematics(self, render_id=None):
        # The goal is to compute the pose to update the shape and the render
        # However for
        idx = 0
        if self.rigid_body is not None:
            rigid_body_pose = self.rigid_body.fk()
        for obj, shape, visual in zip(self.init_objects, self.shapes, self.visuals):
            if not isinstance(obj, RigidBody):
                pose = obj.fk()
            else:
                pose = rigid_body_pose[:, idx]; idx += 1

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
        # update
        self.mechanism(self.gravity, self.collision, tau=None, wrench=None)
        # we may need to add toi
        qacc_obj, qacc_art = self.mechanism.solve(self.dt)

        if self.rigid_body is not None:
            self.rigid_body.euler_(qacc_obj, self.dt)

        if self.articulation is not None:
            self.articulation.euler_(qacc_art, self.dt)

    def step(self):
        assert self.rigid_body is not None or self.articulation is not None,\
            "please add at least one object or reset before running"
        # perhaps we have to apply force
        for i in range(self.frameskip):
            self.do_simulation()
