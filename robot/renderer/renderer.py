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

class Shape:
    def set_pose(self, pose):
        raise NotImplementedError

    def off(self):
        raise NotImplementedError


class RigidBody(Shape):
    def __init__(self, scene, tm, pose=np.eye(4)):
        # tm: trimesh.mesh
        smooth = tm.visual.face_colors is None # xjb hack
        self.tm = tm # we preserve tm for load and save

        import pyrender
        mesh = pyrender.Mesh.from_trimesh(tm, smooth=smooth)
        self.scene = scene
        self.local_pose = pose
        self.node = pyrender.Node(mesh=mesh, matrix=pose)
        self._last_pose = None
        self.on()

    def set_pose(self, pose):
        # the input is a matrix
        if self.scene.has_node(self.node):
            self.scene.set_pose(self.node, pose=pose @ self.local_pose)
            self._last_pose = None
        else:
            self._last_pose = pose

    def get_pose(self):
        return self.node.matrix

    def on(self):
        if not self.scene.has_node(self.node):
            self.scene.add_node(self.node)
            if self._last_pose is not None:
                self.scene.set_pose(self.node, pose=self._last_pose @ self.local_pose)
                self._last_pose = None

    def off(self):
        if self.scene.has_node(self.node):
            self.scene.remove_node(self.node)


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
        mesh = trimesh.primitives.Box(extents=np.array(size))
        mesh.visual.vertex_colors = color
        super(Box, self).__init__(scene, mesh, pose)


class Compose(Shape):
    def __init__(self, *args):
        self.shape = list(args)

    def add_shapes(self, shape):
        # Note the pose is in the local frame
        self.shape.append(shape)

    def set_pose(self, pose):
        for a in self.shape:
            a.set_pose(pose)

    def on(self):
        for a in self.shape:
            a.on()

    def off(self):
        for a in self.shape:
            a.off()

def vec2pose(vec):
    l = np.linalg.norm(vec)
    a, b = np.array([1, 0, 0]), vec / l
    # find quat such that qmult(quat, [1, 0, 0]) = vec
    import transforms3d
    if np.linalg.norm(a - b) < 1e-6:
        pose = np.eye(3)
    elif np.linalg.norm(a + b) < 1e-6:
        pose = transforms3d.quaternions.quat2mat(np.array([0, 0, 0, 1]))
    else:
        v = np.cross(a, b)  # rotation along v
        theta = np.arccos(np.dot(a, b))
        pose = transforms3d.axangles.axangle2mat(v, theta)
    return pose

def Line(scene, start, end, radius, color):
    start = np.array(start)
    end = np.array(end)
    vec = start - end
    l = np.linalg.norm(start-end)
    matrix = np.eye(4)
    matrix[:3,:3] = vec2pose(vec)
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



class Arm(Compose):
    # currently we only consider the robot arm
    # we will consider the general articulator later

    def __init__(self, scene, M, A):
        self.scene = scene
        self.M = totensor(M)[None,:]
        self.A = totensor(A)[None,:]
        self.shape = []
        self.set_q = None

    def fk(self, q):
        q = totensor(q)[None,:]
        return fk_in_space(q, self.M, self.A)[0].detach().cpu().numpy() # (n+1, 4, 4)

    def set_pose(self, q):
        # Notice that set pose is different to set qpos
        Ts = self.fk(q)
        for T, shape in zip(Ts, self.shape):
            if shape is None:
                continue
            shape.set_pose(T)

    def update(self, M, A):
        self.M = totensor(M)[None, :]
        self.A = totensor(A)[None, :]


    def loop(self, K=20, start=-np.pi/3):
        dof = self.A.shape[1]
        a = np.zeros((dof,))
        #b = np.random.random((7,)) * np.pi * 2
        for i in range(dof):
            b = a.copy()
            kk = -start

            a[i] = kk
            b[i] = np.pi *2 + kk

            for j in range(1):
                for k in range(K):
                    q = a + (b-a)/K * k
                    self.set_pose(q)
                    yield q
            a[i] = np.pi * 2 + kk


class ScrewArm(Arm):
    def __init__(self, scene, M, A, G=None, scale=0.1, axis_scale=0.1):
        super(ScrewArm, self).__init__(scene, M, A)

        if G is not None:
            self.G = totensor(G)[None, :]
        self.links = []

        for screw in A:
            #inertia, mass = g[[0, 1, 2], [0, 1, 2]], g[3, 3]

            cmass_box = Box(scene, (scale * 0.7, scale * 0.7, scale * 0.7), (0, 255, 0, 255), np.eye(4))
            cmass_axis = Axis(scene, np.eye(4), scale=axis_scale)
            # visualize cmass...

            pose = self.screw_pose(screw)
            screw = Cylinder(scene, height=scale * 1.5, radius=scale * 0.3, color=(255, 0, 0, 255), pose=pose)
            self.links.append(Compose(cmass_box, cmass_axis, screw))

        self.links.append(Sphere(scene, (0,0,0), scale * 0.7, (0, 0, 255)))

    def screw_pose(self, screw):
        # w, q -> screw=(w,-wxq)
        # q = o - p, where <o-p,w>=90, p=cmass
        w, wxq = screw[:3], -screw[3:]
        # self.screw.append()
        q = np.cross(wxq, w)
        if np.linalg.norm(wxq) > 1e-16:
            t = np.array([w, wxq / np.linalg.norm(wxq), q / np.linalg.norm(q)]).T
        else:
            t, q = vec2pose(w), np.zeros((3,))
        pose = np.eye(4)
        pose[:3, :3] = t
        pose[:3, 3] = q
        return pose

    def update(self, M, A):
        super(ScrewArm, self).update(M, A)
        for link, a in zip(self.links, A):
            assert isinstance(link.shape[-1], Cylinder)
            link.shape[-1].local_pose = self.screw_pose(a)

    def set_pose(self, q):
        super(ScrewArm, self).set_pose(q)
        Ts = self.fk(q)
        for T, link in zip(Ts, self.links):
            link.set_pose(T)

    def on(self):
        super(ScrewArm, self).on()
        for a in self.links:
            a.on()

    def off(self):
        super(ScrewArm, self).off()
        for a in self.links:
            a.off()


class Renderer:
    # the renderer it self doesn't do the resource management
    def __init__(self, camera_pose=np.eye(4), ambient_light=(0.5, 0.5, 0.5),
                 bg_color=(0, 0, 0)):
        #This is not necessasry for Teamviewer of dummy x videos..
        #import pyglet
        #pyglet.options['shadow_window'] = False
        # look towards x coordinates
        # y is on the left for the right-hand coordiates

        import pyrender
        self.scene = pyrender.Scene(ambient_light=ambient_light, bg_color=bg_color)
        self._viewers = OrderedDict()

        #camera = pyrender.(yfov=1.1, aspectRatio=1)
        #camera = pyrender.OrthographicCamera()
        camera = pyrender.PerspectiveCamera(yfov=1.4, znear=0.001)#, aspectRatio=1.414)
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

    def delete_rgb_viewer(self):
        _viewer = self._viewers.get('rgb_array')
        if _viewer is not None:
            # 我能怎么办啊。。我也很绝望啊。。。
            _viewer.delete()
            self._viewers['rgb_array'] = None

    def _get_viewer(self, mode, width=500, height=500):
        import pyrender
        _viewer = self._viewers.get(mode)
        if _viewer is None:
            if mode == 'human' or mode == 'interactive':
                self.delete_rgb_viewer()

            if mode == 'rgb_array':
                _viewer = self._viewers.get('human')
                if _viewer is not None:
                    _viewer.render_lock.release()
                    _viewer.close_external()
                    while _viewer.is_active:
                        pass
                self._viewers['human'] = None

            if mode == 'human':
                class DaemonViewer(pyrender.Viewer):
                    def __init__(self, *args, **kwargs):
                        super(DaemonViewer, self).__init__(*args, **kwargs, run_in_thread=False)

                        from threading import Thread
                        self._thread = Thread(target=self._init_and_start_app2)
                        self._thread.daemon = True # some times it will block.. very strange
                        # I should never try to solve all opengl problems because they are tooooooo stupid....
                        self._run_in_thread = True
                        self._thread.start()

                    def _init_and_start_app(self):
                        pass

                    def _init_and_start_app2(self):
                        super(DaemonViewer, self)._init_and_start_app()

                _viewer = DaemonViewer(self.scene)
                _viewer.render_lock.acquire()
            elif mode == 'interactive':
                _viewer = pyrender.Viewer(self.scene)
            elif mode == 'rgb_array':
                _viewer = pyrender.OffscreenRenderer(width, height)
            else:
                raise NotImplementedError(f"viewer mode {mode} is not implemented")

            self._viewers[mode] = _viewer
        return _viewer

    def render(self, mode='rgb_array', width=500, height=500):
        _viewer = self._get_viewer(mode, width, height)
        if mode == 'rgb_array':
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
        assert name not in self._objects, f"{name} already exists in the renderer"
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

    def screw_arm(self, M, A, scale=0.1, name=None):
        return self.register(ScrewArm(self.scene, M, A, None, scale=scale), name)

    def identity(self):
        return np.eye(4)

    def x2y(self):
        raise NotImplementedError("I think the third column should be (0, 0, -1) to make it to be right-hand, but I do not have the time to check it")
        return np.array([[0, 1, 0, 0],
                         [1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1],], dtype=np.float32)

    def x2z(self):
        raise NotImplementedError("I think the third column should be (-1, 0, 0) to make it to be right-hand, but I do not have the time to check it")
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
        pose[:3,3] = p
        return pose

    def save(self, path):
        #for k in self._viewers.values():
            # destroy _viewers for pickle
        tmp = self._viewers

        self._viewers = OrderedDict()
        print('save...')
        with open(path, 'wb') as f:
            pickle.dump(self, f)

        self._viewers = tmp

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            r = pickle.load(f)

        for i in r.scene.nodes:
            if i.mesh is not None:
                for j in i.mesh.primitives:
                    j.delete()
        return r

    def get(self, name):
        return self._objects[name]
