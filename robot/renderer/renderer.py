# visualization tool for shapes
# need to follow pyrender's documentation to set up the off-sceen render
# Ok, the renderer is actually highly correlated with the forward kinematics in the robots ..,
# this is why most physical simulator has their own render for fast speed and reduced resource
# however, for research purpose, I am going to write a pure python forward kinematics module.
# This would be much easier for us to debug

# just like in a physical simulator, we want to represent the system in generalized coordinates

import os
try:
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa' # remove this for human mode ..
    import pyrender
except ImportError:
    del os.environ['PYOPENGL_PLATFORM']
    import pyrender

import time
import trimesh
import numpy as np
import transforms3d
from collections import OrderedDict
from .arm_visual import Arm


class RigidBody:
    def __init__(self, scene, tm, material=None, pose=np.eye(4)):
        # tm: trimesh.mesh
        smooth = tm.visual.face_colors is None # xjb hack

        mesh = pyrender.Mesh.from_trimesh(tm, smooth=smooth)
        self.scene = scene
        self.node = pyrender.Node(mesh=mesh, matrix=pose)
        self.scene.add_node(self.node)

    def set_pose(self, pose):
        # the input is a matrix
        self.scene.set_pose(self.node, pose=pose)

    def get_pose(self):
        return self.node.matrix


class Sphere(RigidBody):
    # we can visit the scene
    def __init__(self, scene, center, radius, color, material=None, subdivisions=2):
        mesh = trimesh.primitives.Sphere(radius=radius,
                                         center=(0, 0, 0),
                                         subdivisions=subdivisions)
        mesh.visual.face_colors = color

        pose = np.eye(4)
        pose[:3, 3] = center
        super(Sphere, self).__init__(scene, mesh, material, pose)

    def set_center(self, pos):
        t = np.eye(4);t[:3,3] = pos
        self.set_pose(t)

class Compose:
    def __init__(self):
        self.shape = []
        self.local_pose = []

    def add_shapes(self, shape, local_pose):
        # Note the pose is in the local frame
        self.shape.append(shape)
        self.local_pose.append(local_pose)

    def set_pose(self, pose):
        for a, b in zip(self.shape, self.local_pose):
            a.set_pose(pose @ b)


class Renderer:
    # the renderer it self doesn't do the resource management
    # e.g., the renderer doesn't use a dict to maintain all objects in the scene
    #    because it has already be maintained by the pyrender.Scene already

    def __init__(self, camera_pose=np.eye(4), ambient_light=(0.02, 0.02, 0.02), bg_color=(32, 32, 32)):
        self.scene = pyrender.Scene(ambient_light=ambient_light, bg_color=bg_color)

        self._viewers = OrderedDict()

        #camera = pyrender.PerspectiveCamera(yfov=1.1, aspectRatio=1)
        camera = pyrender.PerspectiveCamera(yfov=1.4)#, aspectRatio=1.414)
        self.camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        self.scene.add_node(self.camera_node)
        self._set_camera_pose(camera_pose)

    def set_pose(self, a, pose):
        if isinstance(a, pyrender.Node):
            self.scene.set_pose(a, pose)
        else:
            a.set_pose(pose)

    def _set_camera_pose(self, matrix):
        # pyrender: right angle, look to -z axis, x is the right
        """
        print('x', matrix@np.array([1, 0, 0, 1]))
        print('y', matrix@np.array([0, 1, 0, 1]))
        print('z', matrix@np.array([0, 0, 1, 1]))
        """
        t = matrix.copy()
        t[:3,:3] =matrix[:3,:3] @ np.array(
            [[0, 0, -1],
             [-1, 0, 0],
             [0, 1, 0]]
        )
        """
        print('x', t@np.array([0, 0, -1, 1]))
        print('y', t@np.array([-1, 0, 0, 1]))
        print('z', t@np.array([0, 1, 0, 1]))
        print("================")
        """
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
        #pose = self.actor.pose
        #pose = Pose(np.array([x, y, z]), pose.q)
        matrix = self._get_camera_pose()
        matrix[:3,3] = np.array([x, y, z])
        self._set_camera_pose(matrix)

    def set_camera_rotation(self, yaw, pitch):
        #yaw += 1.57 * 4
        mat = transforms3d.euler.euler2mat(0, -pitch, yaw)
        matrix = self._get_camera_pose()

        t = np.eye(4)
        t[:3,3] = matrix[:3,3]
        t[:3,:3] = mat @ matrix[:3,:3]
        self._set_camera_pose(t)

    def add_point_light(self, position, color):
        light = pyrender.PointLight(color=color, intensity=18.0)
        t = np.eye(4)
        t[:3,3] = position
        self.scene.add(light, pose=t)

    def _get_viewer(self, mode, width=500, height=500):
        _viewer = self._viewers.get(mode)
        if _viewer is None:
            if mode == 'human':
                _viewer = pyrender.Viewer(self.scene) # , run_in_thread=True
            elif mode == 'rgb_array':
                _viewer = pyrender.OffscreenRenderer(width, height)
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
        else:
            #_viewer.render_lock.release()
            #time.sleep(1./24) # sleep for 1./24 second for render...
            #_viewer.render_lock.acquire()
            return None

    def make_mesh(self, mesh, pose=np.eye(4)):
        return RigidBody(self.scene, mesh, pose)

    def make_sphere(self, center, r, color, material=None):
        return Sphere(self.scene, center, r, color, material)

    def make_compose(self):
        return Compose()

    def make_arm(self, M, A):
        return Arm(self.scene, M, A)

