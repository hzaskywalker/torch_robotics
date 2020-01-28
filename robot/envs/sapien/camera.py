import numpy as np
from typing import Union
import sapien.core as sapien_core
from sapien.core import Pose
import transforms3d


class CameraRender:
    def __init__(self, scene, name, width: int, height: int, fov=1.1, near=0.01, far=100):
        builder = scene.create_actor_builder()
        actor = builder.build(True, str(name))

        camera = scene.add_mounted_camera(name, actor, sapien_core.Pose([0, 0, 0], [1, 0, 0, 0]), width, height, fov,
                                        fov, near, far)

        self.actor = actor
        self.camera = camera

    def render(self):
        self.camera.take_picture()
        return self.camera.get_color_rgba()

    def set_camera_position(self, x, y, z):
        pose = self.actor.pose
        pose = Pose(np.array([x, y, z]), pose.q)
        self.actor.set_pose(pose)

    def set_camera_rotation(self, yaw, pitch):
        quat = transforms3d.euler.euler2quat(yaw, pitch, 0)
        pose = self.actor.pose
        q = pose.q
        pose = Pose(pose.p, transforms3d.quaternions.qmult(quat, q))
        self.actor.set_pose(pose)

    def set_current_scene(self, scene):
        pass
