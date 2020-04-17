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


class RigidBody:
    def __init__(self, scene, tm, material=None, pose=np.eye(4)):
        # tm: trimesh.mesh

        mesh = pyrender.Mesh.from_trimesh(tm)
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
    def __init__(self, scene, center, radius, color, material=None, subdivisions=3):
        mesh = trimesh.primitives.Sphere(radius=radius,
                                         center=(0, 0, 0),
                                         subdivisions=subdivisions)
        for facet in mesh.facets:
            mesh.visual.face_colors[facet] = color

        pose = np.eye(4)
        pose[:3, 3] = center
        super(Sphere, self).__init__(scene, mesh, material, pose)


class RobotArm:
    # articulator is also a kind of shape
    def __init__(self):
        pass


class Renderer:
    def __init__(self, camera_pose=np.eye(4)):
        self.scene = pyrender.Scene(ambient_light=[0.02, 0.02, 0.02], bg_color=[0., 0., 0.])

        self._viewers = OrderedDict()

        # right angle, look to -z axis, x is the right
        #camera = pyrender.PerspectiveCamera(yfov=1.1, aspectRatio=1)
        camera = pyrender.PerspectiveCamera(yfov=1.5)
        self.camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        self.scene.add_node(self.camera_node)

    def set_pose(self, a, pose):
        if isinstance(a, pyrender.Node):
            self.scene.set_pose(a, pose)
        else:
            a.set_pose(pose)

    def set_camera_position(self, x, y, z):
        #pose = self.actor.pose
        #pose = Pose(np.array([x, y, z]), pose.q)
        matrix = self.camera_node.matrix
        matrix[:3,3] = np.array([x, y, z])
        self.set_pose(self.camera_node, pose=matrix)

    def set_camera_rotation(self, yaw, pitch):
        #yaw += 1.57 * 4
        mat = transforms3d.euler.euler2mat(0, -pitch, yaw)
        matrix = self.camera_node.matrix

        t = np.eye(4)
        t[:3,3] = matrix[:3,3]
        t[:3,:3] = mat @ matrix[:3,:3]
        self.set_pose(self.camera_node, pose=t)

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
            color, depth = _viewer.render(self.scene)
            return color[...,::-1]
        else:
            #_viewer.render_lock.release()
            #time.sleep(1./24) # sleep for 1./24 second for render...
            #_viewer.render_lock.acquire()
            return None

    def add_sphere(self, center, r, color, material=None):
        return Sphere(self.scene, center, r, color, material)

    def add_articulator(self):
        pass

