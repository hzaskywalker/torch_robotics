import numpy as np
import sapien
import sapien.core as sapien_core
import os
from .gameplay.simulator import Simulator, PxIdentity, x2y, add_link, Pose
from .gameplay import Constraint

def get_assets_path() -> str:
    #root = get_project_root()
    root = '/home/hza/physx_simulation/'
    if not os.path.exists(root):
        raise FileExistsError(f"NO file {root}")
    return os.path.abspath(os.path.join(root, "assets"))


def read_part_mobility(scene: sapien_core.Scene, id, kinematics=True):
    urdf_file = sapien.asset.download_partnet_mobility(id)
    urdf_loader:sapien_core.URDFLoader = scene.create_urdf_loader()
    urdf_loader.scale = 0.8
    urdf_loader.default_density = 10000
    urdf_loader.fix_root_link = False
    if kinematics:
        return urdf_loader.load_kinematic(urdf_file)
    else:
        return urdf_loader.load(urdf_file)


class Sim3D(Simulator):
    def __init__(self):
        super(Sim3D, self).__init__()
        self.register('move_xyz', MoveXYZ)
        # for articulator, we have two special things: the actuator and the ee_idx, they defines the functionality of the agents

    def build_scene(self):
        self.scene.add_ground(0.)

        movo_material = self.sim.create_physical_material(3.0, 2.0, 0.01)
        self.agent = self._load_robot('agent', "all_robot", movo_material)

        self.table: sapien_core.Articulation = read_part_mobility(self.scene, 24931)
        self.table_pos = Pose([1.2, 0, 0.5], x2y)
        actor_builder = self.scene.create_actor_builder()

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
            assert joint.get_dof() == 1
            qname = joint.name
            flag = 'right' in qname or 'gripper' in qname
            if not flag and joint.get_dof() > 0:
                self._fixed_joint.append(dof_count)
                self._fixed_value.append(0)
            else:
                self.add_force_actuator(name, dof_count, -50, 50, joint_id)
            dof_count += joint.get_dof()

        for idx, i in enumerate(agent.get_links()):
            if i.name == 'right_gripper_base_link':
                self._ee_link_idx[name] = idx
                break

        initial_qpos = np.array([0., -1.381, 0, 0.05, -0.9512, 0.387, 0.608, 2.486, 0.986, 0.986, 0.986, 0., 0.])
        agent.set_qpos(initial_qpos)

        return agent

    @property
    def gripper(self):
        return self.agent.get_links()[self._ee_link_idx['agent']]

    def step_scene(self):
        self.table.set_root_pose(self.table_pos)
        self.agent.set_qf([0.001] * self.agent.dof)
        super(Sim3D, self).step_scene()

        q = self.agent.get_qpos()
        q[self._fixed_joint] = self._fixed_value
        self.agent.set_qpos(q)

        self.table.set_qpos([-0.150])

    def build_renderer(self):
        self.scene.set_ambient_light([.4, .4, .4])
        self.scene.set_shadow_light([1, -1, -1], [.5, .5, .5])
        self.scene.add_point_light([2, 2, 2], [1, 1, 1])
        self.scene.add_point_light([2, -2, 2], [1, 1, 1])
        self.scene.add_point_light([-2, 0, 2], [1, 1, 1])

        self._renderer.set_camera_position(3, -1.5, 1.65)
        self._renderer.set_camera_rotation(-3.14 - 0.5, -0.2)
        self._renderer.set_current_scene(self.scene)


class MoveXYZ(Constraint):
    def __init__(self, xyz, agent=None):
        super(MoveXYZ, self).__init__(priority=1, perpetual=True)
        if isinstance(xyz, tuple) or isinstance(xyz, list):
            xyz = np.array(xyz)
        assert xyz.shape == (3,)
        self.xyz = xyz
        self.model = agent

    def preprocess(self, sim: Sim3D):
        target = self.xyz.copy()  # ensure that we don't change the action outside of this scope
        #rot_ctrl = np.array([1, 0, 0, 0])
        if self.model is None:
            name = 'agent'
            model = sim.objects[name]
        else:
            model = self.model
            name = model.get_name()

        ee_idx = sim._ee_link_idx[name]
        pos_ctrl = (target - model.get_links()[ee_idx].pose.p[:3])

        joints = model.get_joints()
        jac = model.compute_jacobian()[ee_idx*6:ee_idx*6+6] # in joint space
        actuator_idx = sim._actuator_dof_idx[name]
        jac = jac[:3, actuator_idx]

        delta = np.linalg.lstsq(jac, pos_ctrl)[0] # in joint space
        targets = model.get_qpos()[actuator_idx] + delta * 20

        for target, index in zip(targets, sim._actuator_joitn_idx[name]):
            joints[index].set_drive_property(10000, 10000)
            joints[index].set_drive_target(target)

        qf = model.compute_passive_force()
        model.set_qf(qf)
