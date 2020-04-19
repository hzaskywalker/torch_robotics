# visualization tool for shapes
# need to follow pyrender's documentation to set up the off-sceen render
# Ok, the renderer is actually highly correlated with the forward kinematics in the robots ..,
# this is why most physical simulator has their own render for fast speed and reduced resource
# however, for research purpose, I am going to write a pure python forward kinematics module.
# This would be much easier for us to debug

# just like in a physical simulator, we want to represent the system in generalized coordinates

import os
import trimesh
import numpy as np
import pickle

import transforms3d
from collections import OrderedDict
from ..torch_robotics import totensor, fk_in_space


class RigidBody:
    def __init__(self, scene, tm, pose=np.eye(4)):
        # tm: trimesh.mesh
        smooth = tm.visual.face_colors is None # xjb hack
        self.tm = tm # we preserve tm for load and save

        import pyrender
        mesh = pyrender.Mesh.from_trimesh(tm, smooth=smooth)
        self.scene = scene
        self.local_pose = pose
        self.node = pyrender.Node(mesh=mesh, matrix=pose)
        self.scene.add_node(self.node)

    def set_pose(self, pose):
        # the input is a matrix
        self.scene.set_pose(self.node, pose=pose @ self.local_pose)

    def get_pose(self):
        return self.node.matrix


class Sphere(RigidBody):
    # we can visit the scene
    def __init__(self, scene, center, radius, color, subdivisions=2):
        mesh = trimesh.primitives.Sphere(radius=radius,
                                         center=(0, 0, 0),
                                         subdivisions=subdivisions)
        mesh.visual.vertex_colors = color

        pose = np.eye(4)
        pose[:3, 3] = center
        super(Sphere, self).__init__(scene, mesh, pose)


class Cylinder(RigidBody):
    def __init__(self, scene, height, radius, color, pose, sections=30):
        x2z = np.array([[0, 0, 1, 0],
         [0, 1, 0, 0],
         [1, 0, 0, 0],
         [0, 0, 0, 1]])
        mesh = trimesh.primitives.Cylinder(height=height,
                                         radius=radius,
                                         sections=sections,
                                         transform=x2z)
        mesh.visual.vertex_colors = color
        super(Cylinder, self).__init__(scene, mesh, pose)


class Capsule(RigidBody):
    def __init__(self, scene, height, radius, color=None, pose=np.eye(4), sections=32):
        mesh = trimesh.primitives.Capsule(height=height, radius=radius, sections=sections, transform=np.array(
            [[0, 0, 1, -height/2],
             [0, 1, 0, 0],
             [1, 0, 0, 0],
             [0, 0, 0, 1]]
        ))
        if color is None:
            color = np.random.randint(0, 255, size=(3,))
        mesh.visual.vertex_colors = color
        super(Capsule, self).__init__(scene, mesh, pose)


class Box(RigidBody):
    def __init__(self, scene, size, color=None, pose=np.eye(4)):
        if color is None:
            color = np.random.randint(0, 255, size=(3,))
        mesh = trimesh.primitives.Box(extents=size)
        mesh.visual.vertex_colors = color
        super(Box, self).__init__(scene, mesh, pose)


class Compose:
    def __init__(self, *args):
        self.shape = list(args)

    def add_shapes(self, shape):
        # Note the pose is in the local frame
        self.shape.append(shape)

    def set_pose(self, pose):
        for a in self.shape:
            a.set_pose(pose)


def Line(scene, start, end, radius, color):
    start = np.array(start)
    end = np.array(end)
    vec = start - end
    l = np.linalg.norm(start-end)

    a, b = np.array([1, 0, 0]), vec / l
    # find quat such that qmult(quat, [1, 0, 0]) = vec
    import transforms3d
    if np.linalg.norm(a - b) < 1e-6:
        pose = np.eye(4)
    elif np.linalg.norm(a + b) < 1e-6:
        pose = transforms3d.quaternions.quat2mat(np.array([0, 0, 0, 1]))
    else:
        v = np.cross(a, b)  # rotation along v
        theta = np.arccos(np.dot(a, b))
        pose = transforms3d.axangles.axangle2mat(v, theta)
    matrix = np.eye(4)
    matrix[:3,:3] = pose
    matrix[:3, 3] = (start+end)/2
    return Capsule(scene, l, radius, color, pose=matrix)


def Axis(scene, pose, scale=1., radius=None, xyz=None):
    if xyz is None:
        xyz = (scale, scale, scale)
    if radius is None:
        radius = max(xyz) * 0.03

    start = pose[:3, 3]
    return Compose(
        Line(scene, start, (pose @ np.array([xyz[0], 0, 0, 1]))[:3], radius, (255, 0, 0)),
        Line(scene, start, (pose @ np.array([0, xyz[1], 0, 1]))[:3], radius, (0, 255, 0)),
        Line(scene, start, (pose @ np.array([0, 0, xyz[2], 1]))[:3], radius, (0, 0, 255))
    )


class Arm:
    # currently we only consider the robot arm
    # we will consider the general articulator later

    def __init__(self, scene, M, A):
        self.scene = scene
        self.M = totensor(M)[None,:]
        self.A = totensor(A)[None,:]
        self.shapes = []

    def add_shapes(self, shape):
        # Note the pose is in the local frame
        self.shapes.append(shape)

    def fk(self, q):
        q = totensor(q)[None,:]
        return fk_in_space(q, self.M, self.A)[0].detach().cpu().numpy() # (n+1, 4, 4)

    def set_pose(self, q):
        # Notice that set pose is different to set qpos
        Ts = self.fk(q)
        for T, shape in zip(Ts, self.shapes):
            if shape is None:
                continue
            shape.set_pose(T)


class ScrewArm(Arm):
    def __init__(self, scene, M, A, G=None, scale=0.1, axis_scale=0.1):
        super(ScrewArm, self).__init__(scene, M, A)

        if G is not None:
            self.G = totensor(G)[None, :]
        self.links = []

        for screw in A:
            #inertia, mass = g[[0, 1, 2], [0, 1, 2]], g[3, 3]

            cmass = [Box(scene, (scale * 0.7, scale * 0.7, scale * 0.7), (0, 255, 0, 255), np.eye(4)),
                     Axis(scene, np.eye(4), scale=axis_scale)
                 ]
            # visualize cmass...

            # w, q -> screw=(w,-wxq)
            # q = o - p, where <o-p,w>=90, p=cmass
            w, wxq = screw[:3], -screw[3:]
            #self.screw.append()
            q = np.cross(wxq, w)
            t = np.array([w, wxq/np.linalg.norm(wxq), q/np.linalg.norm(q)]).T
            pose = np.eye(4)
            pose[:3,:3] = t
            pose[:3, 3] = q

            screw = [
                #Sphere(scene, q, scale * 0.7, (255, 0, 255, 255)),
                #Axis(scene, pose, axis_scale*3)
                Cylinder(scene, height=scale * 1.5, radius=scale*0.3, color=(255, 0, 0, 255), pose=pose)
            ]
            self.links.append(Compose(*cmass, *screw))

    def set_pose(self, q):
        super(ScrewArm, self).set_pose(q)
        Ts = self.fk(q)
        for T, link in zip(Ts, self.links):
            link.set_pose(T)


class Renderer:
    # the renderer it self doesn't do the resource management
    # e.g., the renderer doesn't use a dict to maintain all objects in the scene
    #    because it has already be maintained by the pyrender.Scene already

    def __init__(self, camera_pose=np.eye(4), ambient_light=(0.5, 0.5, 0.5),
                 bg_color=(0, 0, 0)):
        import pyglet
        pyglet.options['shadow_window'] = False

        import pyrender
        self.scene = pyrender.Scene(ambient_light=ambient_light, bg_color=bg_color)
        self._viewers = OrderedDict()

        #camera = pyrender.(yfov=1.1, aspectRatio=1)
        #camera = pyrender.OrthographicCamera()
        camera = pyrender.PerspectiveCamera(yfov=1.4, znear=0.005)#, aspectRatio=1.414)
        self.camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        self.scene.add_node(self.camera_node)
        self._set_camera_pose(camera_pose)

        self._objects = {}

    def set_pose(self, a, pose):
        import pyrender
        if isinstance(a, pyrender.Node):
            self.scene.set_pose(a, pose)
        else:
            a.set_pose(pose)

    def _set_camera_pose(self, matrix):
        # pyrender: right angle, look to -z axis, x is the right
        t = matrix.copy()
        t[:3,:3] =matrix[:3,:3] @ np.array(
            [[0, 0, -1],
             [-1, 0, 0],
             [0, 1, 0]]
        )
        self.set_pose(self.camera_node, pose=t)

    def _get_camera_pose(self):
        matrix = self.camera_node.matrix.copy()
        matrix[:3,:3] = matrix[:3,:3] @ np.array(
            [[0, 0, -1],
             [-1, 0, 0],
             [0, 1, 0]]
        ).T
        return matrix

    def set_camera_position(self, x, y, z):
        matrix = self._get_camera_pose()
        matrix[:3,3] = np.array([x, y, z])
        self._set_camera_pose(matrix)

    def set_camera_rotation(self, yaw, pitch):
        # so basically, first rotate pitch about y (up: positive, down: negative)
        # then rotate yaw about the z (right: positive, left: negative)
        mat = transforms3d.euler.euler2mat(0, -pitch, yaw)
        matrix = self._get_camera_pose()

        t = np.eye(4)
        t[:3,3] = matrix[:3,3]
        t[:3,:3] = mat
        self._set_camera_pose(t)

    def add_point_light(self, position, color, intensity=18.0):
        import pyrender

        light = pyrender.PointLight(color=color, intensity=intensity)
        t = np.eye(4)
        t[:3,3] = position
        self.scene.add(light, pose=t)

    def _get_viewer(self, mode, width=500, height=500):
        import pyrender
        _viewer = self._viewers.get(mode)
        if _viewer is None:
            if mode == 'human':
                _viewer = pyrender.Viewer(self.scene, run_in_thread=True)
                _viewer.render_lock.acquire()
            elif mode == 'interactive':
                _viewer = pyrender.Viewer(self.scene)
            elif mode == 'rgb_array':
                try:
                    _viewer = pyrender.OffscreenRenderer(width, height)
                except AttributeError:
                    raise Exception("export PYOPENGL_PLATFORM=osmesa for off-screen render!!!!")
            else:
                raise NotImplementedError(f"viewer mode {mode} is not implemented")

            self._viewers[mode] = _viewer
        return _viewer

    def render(self, mode='rgb_array', width=500, height=500):
        _viewer = self._get_viewer(mode, width, height)
        if mode == 'rgb_array':
            try:
                color, depth = _viewer.render(self.scene)
            except:
                # TODO: try twice.. very ugly hack as I found I can't use sapien and pyrender together
                _viewer.delete()
                self._viewers[mode] = None
                _viewer = self._get_viewer(mode, width, height)
                color, depth = _viewer.render(self.scene)
            return color[...,::-1]
        elif mode=='human':
            _viewer.render_lock.release()
            import time
            time.sleep(1./24) # sleep for 1./24 second for render...
            _viewer.render_lock.acquire()
        else:
            self._viewers[mode] = None


    def register(self, obj, name):
        if name is None:
            name = f'obj{len(self._objects)}'
        assert name not in self._objects
        self._objects[name] = obj
        return obj

    def trimesh(self, mesh, pose=np.eye(4), name=None):
        return self.register(RigidBody(self.scene, mesh, pose=pose), name)

    def sphere(self, center, r, color, name=None):
        return self.register(Sphere(self.scene, center, r, color), name)

    def capsule(self, height, radius, color, pose, name=None):
        return self.register(
            Capsule(self.scene, height, radius, color, pose=pose), name)

    def box(self, size, color, pose, name=None):
        return self.register(Box(self.scene, size, color, pose), name=name)

    def cylinder(self, height, radius, color, pose, name=None):
        return self.register(Cylinder(self.scene, height, radius, color, pose), name=name)

    def compose(self, *args, name=None):
        out = self.register(Compose(), name)
        for i in args:
            out.add_shapes(i)
        return out

    def line(self, start, end, radius, color, name=None):
        return self.register(Line(self.scene, start, end, radius, color), name=name)

    def axis(self, pose, scale=1, radius=None, xyz=None, name=None):
        return self.register(Axis(self.scene, pose, scale, radius, xyz), name=name)

    def make_arm(self, M, A, name=None):
        return self.register(Arm(self.scene, M, A), name)

    def screw_arm(self, M, A, name=None):
        return self.register(ScrewArm(self.scene, M, A, None), name)

    def x2y(self):
        return np.array([[0, 1, 0, 0],
                         [1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1],], dtype=np.float32)

    def identity(self):
        return np.eye(4)

    def x2z(self):
        return np.array([[0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [1, 0, 0, 0],
                         [0, 0, 0, 1],], dtype=np.float32)

    def y2z(self):
        return np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],], dtype=np.float32)

    def translate(self, p):
        pose = np.eye(4)
        pose[:3,3] = pose
        return pose

    def save(self, path):
        for k in self._viewers.values():
            # destroy _viewers for pickle
            k.delete()

        self._viewers = OrderedDict()
        print('save...')
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def get(self, name):
        return self._objects[name]

    def __del__(self):
        if 'human' in self._viewers:
            v = self._viewers['human']
            v.render_lock.release()
            v.close_external()
            while v.is_active:
                pass
