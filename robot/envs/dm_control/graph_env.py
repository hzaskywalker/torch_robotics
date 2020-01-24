import torch
from collections import OrderedDict
import numpy as np
import copy
from .dm_env import DmControlWrapper
from dm_control.mujoco.wrapper.mjbindings import mjlib
from robot.utils.quaternion import qmul, qrot
from ..spaces import Array, Dict, Discrete, Quaternion, Angular6d
from ..spaces.utils import cat
from ...utils import rot6d


class GraphSpace(Dict):
    def __init__(self, n, m, dim_edge, dim_node,
               low_node=1, high_node=None,
               low_edge=1, high_edge=None):
        self.n = n
        self.m = m
        node = Array(low=low_node, high=high_node, shape=(n, dim_node))
        edge = Array(low=low_edge, high=high_edge, shape=(m, dim_edge))
        graph = Discrete(n, shape=(2, m))
        super(GraphSpace, self).__init__(node=node, edge=edge, graph=graph)


from ..extension import ExtensionBase

class Extension(ExtensionBase):
    def __init__(self, n, action_list, output_dim):
        self.action_list = np.array(action_list)
        self.m = output_dim//2

        self.x = slice(0, 3)
        self.dx = slice(3, 6)
        self.w = slice(6, 12)
        self.dw = slice(12, 15)
        self.observation_shape = (n, 15)
        self.derivative_shape = (n, 15)
        self.action_shape = (output_dim,)

    def encode_obs(self, obs):
        return obs

    def encode_action(self, action):
        if isinstance(action, np.ndarray):
            output = np.zeros(action.shape[:-1] + self.action_shape, dtype=np.float32)
        else:
            output = torch.zeros(action.shape[:-1] + self.action_shape, device=action.device)
        cc = np.array(self.action_list) if isinstance(action, np.ndarray) else torch.LongTensor(self.action_list)
        if len(action.shape) == 1:
            output[cc] = action
            output[cc + self.m] = action
        else:
            output[:, cc] = action
            output[:, cc+self.m] = action
        return output


    def add(self, a, b):
        return torch.cat([
            a[..., self.x] + b[..., self.x], # update coordinates
            rot6d.rmul(a[..., self.w], b[..., self.w]),
            a[..., self.dw] + b[..., self.dw],
            a[..., self.dx] + b[..., self.dx],
        ], dim=-1)


    def sub(self, a, b):
        return torch.cat([
            a[..., self.x] - b[..., self.x], # update coordinates
            rot6d.rmul(a[..., self.w], rot6d.inv(b[..., self.w])),
            a[..., self.dw] - b[..., self.dw],
            a[..., self.dx] - b[..., self.dx],
            ], dim=-1)


    def distance(self, state, gt, is_batch=True):
        return ((state[..., self.x] - gt[..., self.x])**2).sum(dim=-1) + \
               rot6d.rdist(state[..., self.w], gt[..., self.w]) + \
               ((state[..., self.dw] - gt[..., self.dw]) ** 2).sum(dim=-1) + \
               ((state[..., self.dx] - gt[..., self.dx]) ** 2).sum(dim=-1)


class GraphDmControlWrapper(DmControlWrapper):
    # pass
    def __init__(self, *args, **kwargs):
        super(GraphDmControlWrapper, self).__init__(*args, **kwargs)

        self.init()

        n, m = self._node.shape[0], self._edge.shape[0]

        self.global_space = GraphSpace(n, m, self._edge.shape[1], self._node.shape[1],
                                       low_edge=10000, low_node=10000)
        # global is always fixed
        self._fixed_graph = OrderedDict(
            node = self._node,
            edge = self._edge,
            graph = self._graph,
        )
        assert self._fixed_graph in self.global_space
        self.observation_space = Array(
            low=np.inf, shape=(n, 3+6+3+3)
        )
        self.extension = Extension(n, self.act2jnt, m)


    def get_global(self):
        return self._fixed_graph

    def onehot(self, x, m):
        if isinstance(x, int):
            a = np.zeros((m,))
            a[x] = 1
            return a
        else:
            return np.array([self.onehot(int(i), m) for i in x])

    def init(self):
        # mjModel.opt.{timestep, gravity, wind, magnetic, density, viscosity, impratio, o margin, o solref, o solimp,
        # collision type (one-hot), enableflags (bit array), disableflags (bit array)}.
        model = self.dmcenv.physics.model
        o = model.opt
        # NOTE: I ignored collision_type now
        parameters = [
            o.timestep, o.gravity, o.wind, o.magnetic, o.density, o.viscosity, o.impratio, o.o_margin, o.o_solref,
            o.o_solimp, o.enableflags, o.disableflags
        ]
        _global = []
        for i in parameters:
            if not isinstance(i, np.ndarray):
                _global.append(i)
            else:
                _global += list(i)
        self._global = np.array(_global)

        self._node = np.concatenate([model.body_mass[:, None], model.body_pos, model.body_quat, model.body_inertia,
                                     model.body_ipos, model.body_iquat], axis=1)

        #print(self.dmcenv.physics.named.data.xpos)
        print('n node', len(self._node))
        print('n joint', len(model.jnt_bodyid))
        print('body parentid', model.body_parentid)
        print('body joint num', model.body_jntnum)

        print('jnt bodyid', model.jnt_bodyid)
        print('jnt type', model.jnt_type)

        self._graph = np.stack(
            [[model.body_parentid[i] for i in model.jnt_bodyid],model.jnt_bodyid]
        )
        print('graph')
        print(self._graph)

        edges = [self.onehot(model.jnt_type, 4), model.jnt_axis, model.jnt_pos, model.jnt_solimp, model.jnt_solref,
                 model.jnt_stiffness, model.jnt_limited, model.jnt_range, model.jnt_margin]
        for i in range(len(edges)):
            if len(edges[i].shape) == 1:
                edges[i] = edges[i][:, None]
        self._edge = np.concatenate(edges, axis=1)
        print(self._edge.shape)

        actuator = [self.onehot(model.actuator_biastype, 4), model.actuator_biasprm,
                    model.actuator_cranklength, model.actuator_ctrllimited, model.actuator_ctrlrange,
                    self.onehot(model.actuator_dyntype, 5), model.actuator_dynprm, model.actuator_forcelimited,
                    model.actuator_forcerange, self.onehot(model.actuator_gaintype, 3), model.actuator_gainprm,
                    # , model.actuator_invweight0
                    model.actuator_gear, model.actuator_length0, model.actuator_lengthrange]
        for i in range(len(actuator)):
            if len(actuator[i].shape) == 1:
                actuator[i] = actuator[i][:, None]
        actuator = np.concatenate(actuator, axis=1)
        act_jnttype = model.actuator_trntype
        act_jntid = model.actuator_trnid
        for trntype in act_jnttype:
            assert trntype == 0

        edge_actuator = np.zeros((self._edge.shape[0], actuator.shape[1] + 1))
        edge_actuator[:, :-1] += actuator.mean(axis=0)

        for act, jnt in enumerate(act_jntid[:, 0]):
            edge_actuator[jnt, :-1] = act
            edge_actuator[jnt, -1] = 1

        self._edge = np.concatenate((self._edge, edge_actuator, edge_actuator[:, 0:1]*0), axis=1) #the last is the sign
        inverse_edge = self._edge.copy()
        inverse_edge[..., -1:] += 1
        self._edge = np.concatenate((self._edge, inverse_edge), axis=0) # add sign
        self._graph = np.concatenate((self._graph, self._graph[::-1]), axis=1)

        self.act2jnt = act_jntid[:, 0].copy()

        for idx in range(1, len(self._node)):
            if model.body_jntnum[idx] == 0:
                # no joint, directly connected to the parent.
                print("WARNING>>> no joint, no connection to the parent")


    def getObservation(self, qua=False):
        phy = self.dmcenv.physics
        name = phy.named.data.xpos.axes.row._names

        vel = np.zeros(6)
        vels = []

        for i in range(len(name)):
            mjlib.mj_objectVelocity(phy.model.ptr, phy.data.ptr, 2, i, vel, False)
            vels.append(vel.copy())

        pos = phy.data.xpos[:]
        if not qua:
            angle = phy.data.xmat[:].reshape(-1, 3, 3)[:, :, :2].reshape(-1, 6)
        else:
            angle = phy.data.xquat[:].copy()

        vels = np.array(vels)
        p, w, v, dw = pos, angle, vels[:, 3:], vels[:, :3]
        return np.concatenate([p, v, w, dw], axis=1)

    def forward(self, u, qua=False):
        phy = self.dmcenv.physics
        np.copyto(phy.data.qpos, u[:self.dq_pos])
        if u.shape[-1] >= self.dq:
            np.copyto(phy.data.qvel, u[self.dq_pos:])
        phy.forward()
        return self.getObservation(qua)

    def to_pos_quat(self, xpos):
        dtype = self.dmcenv.physics.data.qpos.dtype
        pos = xpos[:, :3].astype(dtype)
        #quat = rot6d.r2quat(xpos[:, 3:9]).astype(dtype)
        quat = []
        for i in xpos:
            q = np.zeros((4,), dtype=dtype)
            m = rot6d.rmat(torch.Tensor(i[6:12])).detach().numpy().astype(dtype)
            mjlib.mju_mat2Quat(q, m.reshape(-1))
            quat.append(q)
        return pos, np.stack(quat)

    def recompute_geom(self, state):
        pos, quat = self.to_pos_quat(state)
        dtype = pos.dtype
        pos = torch.Tensor(pos)
        quat = torch.Tensor(quat)

        geom_body = self.dmcenv.physics.model.geom_bodyid
        geom_xpos, geom_xmat = [], []
        for geom_id, body_id in enumerate(geom_body):
            local_pos = self.dmcenv.physics.model.geom_pos[geom_id].copy()
            local_quat = self.dmcenv.physics.model.geom_quat[geom_id].copy()
            local_pos = torch.Tensor(local_pos)
            local_quat = torch.Tensor(local_quat)

            geom_pos = qrot(quat[body_id], local_pos) + pos[body_id]
            geom_quat = qmul(quat[body_id], local_quat)
            geom_xpos.append(geom_pos.detach().numpy())

            mat = np.zeros((9,), dtype)
            mjlib.mju_quat2Mat(mat, geom_quat.detach().numpy().astype(dtype))
            geom_xmat.append(mat)

        geom_xpos = np.array(geom_xpos, dtype=dtype)
        geom_xmat = np.array(geom_xmat, dtype=dtype)
        return geom_xpos, geom_xmat

    def render_state(self, state, mode='rgb_array'):
        # pass
        geom_xpos, geom_xmat = self.recompute_geom(state)
        np.copyto(self.dmcenv.physics.data.geom_xpos, geom_xpos)
        np.copyto(self.dmcenv.physics.data.geom_xmat, geom_xmat)
        return self.render(mode)
