import numpy as np
import os
from .gameplay.simulator import Simulator, Pose, PxIdentity, x2y, add_link, sapien_core
from .gameplay import Constraint

def get_assets_path() -> str:
    #root = get_project_root()
    root = '/home/hza/physx_simulation/'
    if not os.path.exists(root):
        raise FileExistsError(f"NO file {root}")
    return os.path.abspath(os.path.join(root, "assets"))


class Sim3D(Simulator):
    def __init__(self):
        super(Sim3D, self).__init__()

    def build_scene(self):
        self.scene.add_ground(0.)

        movo_material = self.sim.create_physical_material(3.0, 2.0, 0.01)
        self.agent = self._load_robot('agent', "all_robot", movo_material)

    def _load_robot(self, name, urdf_path: str, material: sapien_core.PxMaterial) -> None:
        # By default, the robot will loaded with balanced passive force
        # self.loader.fix_base = True
        if urdf_path.startswith('/'):
            fullpath = urdf_path
        else:
            fullpath = os.path.join(get_assets_path(), "robot", f"{urdf_path}.urdf")

        loader: sapien_core.URDFLoader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        agent: sapien_core.Articulation = loader.load(fullpath, material)
        agent.set_root_pose(sapien_core.Pose([0, 0, 0], [1, 0, 0, 0]))

        self.objects[name] = agent


        joints = agent.get_joints()
        self._fixed_joint, self._fixed_value = [], []
        dof_count, joint_id = 0, 0
        for joint_id, joint in enumerate(joints):
            if joint.get_dof() == 0:
                continue
            qname = joint.name
            flag = 'right' in qname and 'gripper' not in qname
            if not flag and joint.get_dof() > 0:
                self._fixed_joint.append(dof_count)
                self._fixed_value.append(0 if 'gripper' not in qname else 0.986)
            dof_count += joint.get_dof()
        return agent

    def step_scene(self):
        super(Sim3D, self).step_scene()

        q = self.agent.get_qpos()
        q[self._fixed_joint] = self._fixed_value
        self.agent.set_qpos(q)

    def build_renderer(self):
        self.scene.set_ambient_light([.4, .4, .4])
        self.scene.set_shadow_light([1, -1, -1], [.5, .5, .5])
        self.scene.add_point_light([2, 2, 2], [1, 1, 1])
        self.scene.add_point_light([2, -2, 2], [1, 1, 1])
        self.scene.add_point_light([-2, 0, 2], [1, 1, 1])

        self._renderer.set_camera_position(3, -1.5, 1.65)
        self._renderer.set_camera_rotation(-3.14 - 0.5, -0.2)
        self._renderer.set_current_scene(self.scene)
