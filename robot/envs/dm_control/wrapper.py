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


class StateFormat:
    def __init__(self, n, d, dq):
        self.n = n
        self.d = d
        self.dq = dq

        self.x = slice(0, 3)
        self.w = slice(3, 9)
        self.dw = slice(9, 12)
        self.dx = slice(12, 15)

    def decode(self, data):
        s, q = data[..., :-self.dq], data[..., -self.dq:]
        s = s.reshape(*s.shape[:-1], self.n, self.d)
        return s, q

    def encode(self, s, q):
        assert q.shape[-1] == self.dq
        assert s.shape[-1] == self.d
        s = s.reshape(*s.shape[:-2], -1)
        if isinstance(s, torch.Tensor):
            return torch.cat((s, q), dim=-1)
        else:
            return np.concatenate((s, q), axis=-1)

    def add(self, a, b):
        return torch.cat([
            a[..., self.x] + b[..., self.x], # update coordinates
            rot6d.rmul(a[..., self.w], b[..., self.w]),
            a[..., self.dw] + b[..., self.dw],
            a[..., self.dx] + b[..., self.dx],
        ], dim=-1)


    def delete(self, a, b):
        return torch.cat([
            a[..., self.x] - b[..., self.x], # update coordinates
            rot6d.rmul(a[..., self.w], rot6d.inv(b[..., self.w])),
            a[..., self.dw] - b[..., self.dw],
            a[..., self.dx] - b[..., self.dx],
            ], dim=-1)


    def dist(self, state, gt):
        return ((state[..., self.x] - gt[..., self.x])**2).sum(dim=-1) + \
               rot6d.rdist(state[..., self.w], gt[..., self.w]) + \
               ((state[..., self.dw] - gt[..., self.dw]) ** 2).sum(dim=-1) + \
               ((state[..., self.dx] - gt[..., self.dx]) ** 2).sum(dim=-1)


class GraphDmControlWrapper(DmControlWrapper):
    # pass
    def __init__(self, *args, **kwargs):
        super(GraphDmControlWrapper, self).__init__(*args, **kwargs)

        self.init()

        n = len(self.dmcenv.physics.data.xpos)
        d = 3 + 6 + 3 + 3
        dq = len(self.dmcenv.physics.data.qpos) * 2
        self.state_format = StateFormat(n, d, dq)
        high = np.zeros((n*d+dq,)) +np.inf
        self.observation_space = gym.spaces.Box(low=-high, high=high) # not information

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
                raise NotImplementedError

    def get_graph(self):
        return self._graph

    def get_edge_attr(self):
        return self._edge

    def get_node_attr(self):
        return self._node

    def action_to_edge(self):
        # map action to edge
        return self.act2jnt

    def forward(self, x, u):
        # the most import thing is to implement the forward function
        raise NotImplementedError

    def getObservation(self):
        phy = self.dmcenv.physics
        name = phy.named.data.xpos.axes.row._names

        vel = np.zeros(6)
        vels = []

        for i in range(len(name)):
            mjlib.mj_objectVelocity(phy.model.ptr, phy.data.ptr, 2, i, vel, False)
            vels.append(vel.copy())

        pos = phy.data.xpos[:]
        angle = phy.data.xmat[:].reshape(-1, 3, 3)[:, :, :2].reshape(-1, 6)

        return self.state_format.encode(
            np.concatenate((pos, angle, np.stack(vels)), axis=-1),
            np.concatenate((phy.data.qpos.copy(), phy.data.qvel.copy()))
        )

