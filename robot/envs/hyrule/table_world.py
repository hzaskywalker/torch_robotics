import numpy as np
import json
from .gameplay.simulator import Simulator, PxIdentity
from .gameplay.parameters import Parameter
from .sim3d import Sim3D, sapien_core, read_part_mobility, Pose, x2y
import sapien

def id2boundingbox(id):
    urdf_file = sapien.asset.download_partnet_mobility(id)
    with open(f'partnet-mobility-dataset/{id}/bounding_box.json', 'r') as f:
        f = json.load(f)
    return np.stack((f['min'], f['max']))[:, [1, 0, 2]]

def translate_box(box, pose):
    p1 = box[0]
    p2 = box[1]

    p1 = (pose * Pose(p1, PxIdentity)).p
    p2 = (pose * Pose(p2, PxIdentity)).p

    t = np.stack((p1, p2))
    return np.stack((t.min(axis=0), t.max(axis=0)))


class SetQF(Parameter):
    # path_length
    def __init__(self, qf, name):
        assert len(qf.shape) == 2
        super(SetQF, self).__init__(qf)
        self.idx = 0
        self.name = name

    def forward(self, sim):
        #if self.idx == 99:
        #    print(self.data[self.idx])
        sim.objects[self.name].set_qf(self.data[self.idx])
        self.idx += 1

    def update(self, data):
        #data[...,-5:] /= 10
        data = data.copy().reshape(self.data.shape)
        data[:, -5:] /= 10
        super(SetQF, self).update(data)
        self.idx = 0


class TableWorld(Sim3D):
    def __init__(self, objs, map, dir=None, names=None, sim=None):
        self.objs = objs
        self.map = map
        self.dir = dir
        self.names = names

        super(TableWorld, self).__init__(sim=sim)

    def place_object(self, name, instance_id):
        obj_bbx = id2boundingbox(self.objs[instance_id])

        n, m = self.map.shape
        lx, ly, rx, ry = np.inf, np.inf, -np.inf, -np.inf

        cc = 0
        for i in range(n):
            for j in range(m):
                if self.map[i][j] == instance_id+1:
                    lx, ly = min(i, lx), min(j, ly)
                    rx, ry = max(i, rx), max(j, ry)
                    cc += 1
        assert cc > 0, f"please indictate the location of object {name} in the map"

        lower_xy = np.array([lx, ly]) * (self.table_bbox[1] - self.table_bbox[0])[:2]/n + self.table_bbox[0, :2]
        upper_xy = np.array([rx + 1, ry + 1]) * (self.table_bbox[1] - self.table_bbox[0])[:2]/m + self.table_bbox[0, :2]

        lower = obj_bbx[0]
        upper = obj_bbx[1]
        scale = min(min((upper_xy-lower_xy)/(upper[:2]-lower[:2])), 1)
        obj: sapien_core.Articulation = read_part_mobility(self.scene, self.objs[instance_id], scale=scale)
        #print(obj, self.objs[instance_id])
        self.objects[name] = obj

        diff = (upper_xy + lower_xy)/2 - (upper+lower)[:2] * scale/2
        pose = Pose((diff[0], diff[1], self.table_bbox[1, 2]- lower[2] * scale +0.01), PxIdentity)
        obj.set_root_pose(pose)

    def build_scene(self):
        self.scene.add_ground(0.)

        self.timestep = 0

        movo_material = self.sim.create_physical_material(3.0, 2.0, 0.01)

        self.agent = self._load_robot('agent', "all_robot", movo_material)

        self.table_pos = Pose([0.8, 0, 0.25], PxIdentity)
        size = np.array([0.4, 0.4, 0.25])
        self.table = self.add_box(*self.table_pos.p[:2], size, color=(0.5, 0.5, 0.5), name='table', fix=False, density=100000)
        self.objects['table'] = self.table

        self.table_bbox = np.stack((self.table_pos.p - size, self.table_pos.p + size))

        for i in range(len(self.objs)):
            name = f'obj{i}' if self.names is None else self.names[i]
            self.place_object(name, i)


        name = 'box'
        actor_builder = self.scene.create_actor_builder()
        pose = Pose((0.8, 0, 0.5 + 0.02))
        size = np.array((0.02, 0.02, 0.02))
        actor_builder.add_box_visual(Pose(), size, np.array([0, 1, 0]), name)
        actor_builder.add_box_shape(Pose(), size, density=1000)
        box = actor_builder.build(False)
        box.set_pose(pose)
        box.set_name(name)
        self.objects[name] = box


    def add_box(self, x, y, size, color, name, fix=False, density=1000):
        if isinstance(size, int) or isinstance(size, float):
            size = np.array([size, size, size])
        actor_builder = self.scene.create_actor_builder()
        actor_builder.add_box_visual(Pose(), size, color, name)
        actor_builder.add_box_shape(Pose(), size, density=density)
        box = actor_builder.build(fix)

        pos = Pose(np.array((x, y, size[2]+1e-5)))
        box.set_pose(pos)
        box.set_name(name)
        return box

    def step_scene(self):
        #Simulator.step_scene(self)
        self.timestep += 1
        for i in range(3):
            Simulator.step_scene(self)

        q = self.agent.get_qpos()
        q[self._fixed_joint] = self._fixed_value
        self.agent.set_qpos(q)

    def state_vector(self):
        state_vector = super(TableWorld, self).state_vector()
        return np.concatenate(([self.timestep], state_vector), axis=0)

    def load_state_vector(self, vec):
        self.timestep = vec[0]
        super(TableWorld, self).load_state_vector(vec[1:])
        
    def state_dict(self):
        out = super(TableWorld, self).state_dict()
        out['timestep'] = self.timestep
        return out

    def load_state_dict(self, dict):
        super(TableWorld, self).load_state_dict(dict)
        self.timestep = dict['timestep']
