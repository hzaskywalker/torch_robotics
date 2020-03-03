# we want to implement a basic system
import numpy as np
import json
import os
from collections import OrderedDict
import sapien
import sapien.core as sapien_core
from sapien.core import Pose

from .simulator import Simulator
from .waypoints import load_waypoints


PxIdentity = np.array([1, 0, 0, 0])
x2y = np.array([0.7071068, 0, 0, 0.7071068])
x2z = np.array([0.7071068, 0, 0.7071068, 0])


def load_camera(sim, params):
    pass


def get_assets_path() -> str:
    #root = get_project_root()
    root = '/home/hza/physx_simulation/'
    if not os.path.exists(root): raise FileExistsError(f"NO file {root}")
    return os.path.abspath(os.path.join(root, "assets"))


def read_part_mobility(scene: sapien_core.Scene, id, scale=0.8, default_density=10000):
    urdf_file = sapien.asset.download_partnet_mobility(id)
    urdf_loader:sapien_core.URDFLoader = scene.create_urdf_loader()
    urdf_loader.scale = scale
    urdf_loader.default_density = default_density
    urdf_loader.fix_root_link = False
    return urdf_loader.load(urdf_file)


def load_robot(sim: Simulator, name, params: OrderedDict):
    material = sim.sim.create_physical_material(3.0, 2.0, 0.01)
    urdf_path = params.get('urdf', "all_robot")

    # By default, the robot will loaded with balanced passive force
    # self.loader.fix_base = True
    if urdf_path.startswith('/'):
        fullpath = urdf_path
    else:
        fullpath = os.path.join(get_assets_path(), "robot", f"{urdf_path}.urdf")

    loader: sapien_core.URDFLoader = sim.scene.create_urdf_loader()
    loader.fix_root_link = params.get('fixed_root_link', True)
    agent: sapien_core.Articulation = loader.load(fullpath, material)

    robot_pose = params.get('root_pose', {
        'p': [0, 0, 0],
        'q': [1, 0, 0 ,0]
    })
    agent.set_root_pose(sapien_core.Pose(robot_pose['p'], robot_pose['q']))


    actuator = params.get('actuator', [])
    actuator_range = params.get('actuator_range', [])

    lock = params.get('lock', [])
    lock_value = params.get('lock_value', [])

    _lock_dof = [0] * len(lock)
    _lock_value = [0] * len(lock_value)

    _actuator_range = [[0, 2]] * len(actuator)
    _actuator_dof = [0] * len(actuator)
    _actuator_joint = [0] * len(actuator)

    dof_count = 0
    for joint_id, joint in enumerate(agent.get_joints()):
        if joint.get_dof() == 0:
            continue
        assert joint.get_dof() == 1
        qname = joint.name
        if qname in lock:
            for idx, _name in enumerate(lock):
                if _name == qname: lock_id = idx
            _lock_dof[lock_id] = dof_count
            _lock_value[lock_id] = lock_value[lock_id]

        if qname in actuator:
            for idx, _name in enumerate(actuator):
                if _name == qname:
                    actuator_id = idx
            _actuator_range[actuator_id] = actuator_range[actuator_id]
            _actuator_dof[actuator_id] = dof_count
            _actuator_joint[actuator_id] = joint_id

        dof_count += joint.get_dof()


    sim._lock_dof[name] = np.array(_lock_dof)
    sim._lock_value[name] = np.array(_lock_value)
    sim._actuator_range[name] = np.array(_actuator_range)
    sim._actuator_dof[name] = np.array(_actuator_dof)
    sim._actuator_joint[name] = np.array(_actuator_joint)

    ee = params.get('ee', None)
    for idx, i in enumerate(agent.get_links()):
        if i.name == ee:
            sim._ee_link_idx[name] = idx
            break

    initial_qpos = [0., -1.381, 0, 0.05, -0.9512, 0.387, 0.608, 2.486, 0.986, 0.986, 0.986, 0., 0.]
    initial_qvel = [0] * len(initial_qpos)

    initial_qpos = np.array( params.get('qpos', initial_qpos) )
    initial_qvel = np.array( params.get('qvel', initial_qvel) )
    agent.set_qpos(initial_qpos)
    agent.set_qpos(initial_qvel)

    sim.objects[name] = agent
    sim.agent = agent

    return agent


def load_box(sim: Simulator, name, params: OrderedDict):
    # table is fixed objects.. which is not counted a object
    center = params.get('center', [0.8, 0, 0.25])
    size = params.get('size', [0.4, 0.4, 0.25])
    color = params.get('color', (0.5, 0.5, 0.5))
    density = params.get('density', 1000)
    fix = params.get('fix', False)

    actor_builder = sim.scene.create_actor_builder()
    actor_builder.add_box_visual(Pose(), size, color, name)
    actor_builder.add_box_shape(Pose(), size, density=density)
    box = actor_builder.build(fix)

    pos = Pose(center)
    box.set_pose(pos)
    box.set_name(name)

    if not fix:
        sim.objects[name] = box
    return box


def load_scene(sim: Simulator, scene: OrderedDict):
    # ground
    OBJ_TYPE = {
        'robot': load_robot,
        'box': load_box,
    }
    for name, param in scene.items():
        if name == 'ground':
            sim.scene.add_ground(param)
        elif name == 'waypoints':
            sim.cost = load_waypoints(param)
        else:
            OBJ_TYPE[param['type']](sim, name, param)



def load_trajectory():
    pass


def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def dump_json(filepath, params: OrderedDict):
    with open(filepath, 'w') as f:
        return json.dump(params, f)
