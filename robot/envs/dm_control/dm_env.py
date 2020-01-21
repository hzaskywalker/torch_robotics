from gym import core, spaces
from robot.utils import rot6d
import torch
import  numpy as np
import dm_control
from dm_control import suite
from dm_env import specs
from gym.utils import seeding
import gym
from .viewer import DmControlViewer
from dm_control.mujoco.wrapper.mjbindings import mjlib
import numpy as np
import sys
import logging
from robot.utils.quaternion import qmul, qrot


class DmcDiscrete(gym.spaces.Discrete):
    def __init__(self, _minimum, _maximum):
        super().__init__(_maximum - _minimum)
        self.offset = _minimum

def convertSpec2Space(spec, clip_inf=False):
    if spec.dtype == np.int:
        # Discrete
        return DmcDiscrete(spec.minimum, spec.maximum)
    else:
        # Box
        if type(spec) is specs.Array:
            return spaces.Box(-np.inf, np.inf, shape=spec.shape)
        elif type(spec) is specs.BoundedArray:
            _min = spec.minimum
            _max = spec.maximum
            if clip_inf:
                _min = np.clip(spec.minimum, -sys.float_info.max, sys.float_info.max)
                _max = np.clip(spec.maximum, -sys.float_info.max, sys.float_info.max)

            if np.isscalar(_min) and np.isscalar(_max):
                # same min and max for every element
                return spaces.Box(_min, _max, shape=spec.shape)
            else:
                # different min and max for every element
                return spaces.Box(_min + np.zeros(spec.shape),
                                  _max + np.zeros(spec.shape))
        else:
            raise ValueError('Unknown spec!')

def convertOrderedDict2Space(odict):
    if len(odict.keys()) == 1:
        # no concatenation
        return convertSpec2Space(list(odict.values())[0])
    else:
        # concatentation
        numdim = sum([np.int(np.prod(odict[key].shape)) for key in odict])
        return spaces.Box(-np.inf, np.inf, shape=(numdim,))


def convertObservation(spec_obs):
    if len(spec_obs.keys()) == 1:
        # no concatenation
        return list(spec_obs.values())[0]
    else:
        # concatentation
        numdim = sum([np.int(np.prod(spec_obs[key].shape)) for key in spec_obs])
        space_obs = np.zeros((numdim,))
        i = 0
        for key in spec_obs:
            space_obs[i:i+np.prod(spec_obs[key].shape)] = spec_obs[key].ravel()
            i += np.prod(spec_obs[key].shape)
        return space_obs

class DmControlWrapper(gym.Env):

    def __init__(self, domain_name, task_name, task_kwargs=None, visualize_reward=False, render_mode_list=None):

        self.dmcenv:dm_control.suite.control.Environment = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs=task_kwargs,
                                 visualize_reward=visualize_reward)

        # convert spec to space
        self.action_space = convertSpec2Space(self.dmcenv.action_spec(), clip_inf=True)
        self.observation_space = convertOrderedDict2Space(self.dmcenv.observation_spec())

        if render_mode_list is not None:
            self.metadata['render.modes'] = list(render_mode_list.keys())
            self.viewer = {key:None for key in render_mode_list.keys()}
        else:
            self.metadata['render.modes'] = []

        self.render_mode_list = render_mode_list

        # set seed
        self._seed()

    def getObservation(self):
        return convertObservation(self.timestep.observation)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.timestep = self.dmcenv.reset()
        return self.getObservation()

    def _step(self, a):

        if type(self.action_space) == DmcDiscrete:
            a += self.action_space.offset
        self.timestep = self.dmcenv.step(a)

        return self.getObservation(), self.timestep.reward, self.timestep.last(), {}


    def _render(self, mode='human', close=False):

        self.pixels = self.dmcenv.physics.render(**self.render_mode_list[mode]['render_kwargs'])
        if close:
            if self.viewer[mode] is not None:
                self._get_viewer(mode).close()
                self.viewer[mode] = None
            return
        elif self.render_mode_list[mode]['show']:
            self._get_viewer(mode).update(self.pixels)



        if self.render_mode_list[mode]['return_pixel']:

            return self.pixels

    def _get_viewer(self, mode):
        if self.viewer[mode] is None:
            self.viewer[mode] = DmControlViewer(self.pixels.shape[1], self.pixels.shape[0], self.render_mode_list[mode]['render_kwargs']['depth'])
        return self.viewer[mode]


class GraphDmControlWrapper(DmControlWrapper):
    # pass
    def __init__(self, *args, **kwargs):
        super(GraphDmControlWrapper, self).__init__(*args, **kwargs)

        self.init()

        n = len(self.dmcenv.physics.data.xpos)

        from ..spaces import GraphSpace, Array, ArraySpace, Dict
        #d = 3 + 6 + 3 + 3
        #dq = len(self.dmcenv.physics.data.qpos) + len(self.dmcenv.physics.data.qvel)
        #self.state_prior = StatePrior(n, d, dq)
        #self.dq = dq
        #self.dq_pos = len(self.dmcenv.physics.data.qpos)
        #high = np.zeros((n*d+dq,)) +np.inf
        #self.observation_space = gym.spaces.Box(low=-high, high=high) # not information
        n, m = self._node.shape[0], self._edge.shape[0]
        self.observation_space = GraphSpace(n, m, 1, None,
                                       low_edge=10000, low_node=10000)
        #self.action_space = ArraySpace(n, m, 1, 0)
        self.global_space = GraphSpace(n, m, self._edge.shape[1], self._node.shape[1],
                                       low_edge=10000, low_node=10000)

        self._fixed_graph = Dict(
            node = Array(self._node),
            edge = Array(self._edge),
            graph = Array(self._graph),
        ) # global is always fixed
        assert self._fixed_graph in self.global_space

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

        self._edge = np.concatenate((self._edge, edge_actuator), axis=1)
        self.act2jnt = act_jntid[:, 0].copy()

        #print('jnt dof parent', [model.dof_parentid[i] for i in model.jnt_dofadr]) I dont't know what's this
        #print('dof parentid', model.dof_parentid)
        #exit(0)

        for idx in range(1, len(self._node)):
            if model.body_jntnum[idx] == 0:
                # no joint, directly connected to the parent.
                print("WARNING>>> no joint, no connection to the parent")

    """
    def get_graph(self):
        return self._graph

    def get_edge_attr(self):
        return self._edge

    def get_node_attr(self):
        return self._node
        """

    def action_to_edge(self):
        # map action to edge
        return self.act2jnt

    def forward(self, x, u):
        # the most import thing is to implement the forward function
        raise NotImplementedError

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

        return self.state_prior.encode(
            np.concatenate((pos, angle, np.stack(vels)), axis=-1),
            np.concatenate((phy.data.qpos.copy(), phy.data.qvel.copy()))
        )

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
            m = rot6d.rmat(torch.Tensor(i[3:9])).detach().numpy().astype(dtype)
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
