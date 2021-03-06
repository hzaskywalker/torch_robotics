# engine
# support a render and a physical engine ...
# currently we only have an articulation..
# but how can we render the image?
import torch
import numpy as np
import gym
from robot import torch_robotics as tr
from robot.utils import togpu

# Currently I only have the time to implement the 2D simulator?

class Link:
    def __init__(self, T, screw):
        """
        :param name:
        :param T: link frames {i} relative to {i-1} at the home position
        :param screw: (w, v) relative to link frames {i}
        """
        self.T = T
        self.screw = screw
        #theta = np.linalg.norm(screw[:3])

        self._visual = []
        self._inertial = None

    def set_inertial(self, inertial):
        self._inertial = inertial.copy()

    def add_box_visual(self, center, size, color):
        assert len(center) == 3
        assert len(size) == 3
        l, r, t, b = center[0] - size[0], center[0] + size[0], center[1] + size[1], center[1] - size[1]

        self._visual.append(('box', [l, r, t, b], color))

    def add_circle_visual(self, center, radius, color):
        self._visual.append(('circle', [center[0], center[1], radius], color))

    def draw(self, viewer, T):
        for param in self._visual:
            T_shape = T
            if param[0] == 'box':
                (l, r, t, b), color = param[1], param[2]
                shape = viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
                shape.set_color(*color)
            elif param[0] == 'circle':
                (x, y, radius), color = param[1], param[2]
                shape = viewer.draw_circle(radius)
                shape.set_color(*color)
                T_shape = np.dot(T, np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, 0], [0, 0, 0, 1]]))
            else:
                raise NotImplementedError

            theta = np.arctan2(T_shape[1,0], T_shape[0, 0])

            jtrans = viewer.Transform(rotation=theta, translation=(float(T_shape[0,3]), float(T_shape[1, 3])))
            shape.add_attr(jtrans)


class Articulation2D:
    def __init__(self, fixed_base=True,
                 gravity=[0, -9.8, 0],
                 device='cuda:0',
                 max_velocity=100,
                 ftip=None, timestep=0.025):
        self.fixed_base = True
        self._links = []

        self.M = None # home
        self.G = None
        self.A = None
        self.device = device
        self.timestep = timestep
        self.max_velocity = max_velocity
        self.gravity = torch.tensor(gravity, dtype=torch.float64, device=device)
        if ftip is None:
            ftip = np.zeros(6)
        self.ftip = torch.tensor(ftip, dtype=torch.float64, device=device)

        self.qpos = None
        self.qvel = None
        self.qf = None

    def get_qpos(self):
        return self.qpos.clone()

    def get_qvel(self):
        return self.qvel.clone()

    def get_qf(self):
        return self.qf.clone()

    def set_qpos(self, qpos):
        assert qpos.shape[-1] == self.dof, f"qpos.shape[-1] {qpos.shape[-1]}"
        self.qpos = togpu(qpos, torch.float64)

    def set_qvel(self, qvel):
        assert qvel.shape[-1] == self.dof
        self.qvel = togpu(qvel, torch.float64)

    def set_qf(self, qf):
        assert qf.shape[-1] == self.dof, f"qf.shape: {qf.shape}, dof: {self.dof}"
        self.qf = togpu(qf, torch.float64)

    def get_parameters(self, qpos):
        b = qpos.shape[0]
        gravity = self.gravity[None,:].expand(b, -1)
        ftip = self.ftip[None,:].expand(b, -1)
        M = self.M[None,:].expand(b, -1, -1, -1)
        G = self.G[None,:].expand(b, -1, -1, -1)
        A = self.A[None,:].expand(b, -1, -1)
        return gravity, ftip, M, G, A

    def forward_kinematics(self, qpos=None):
        # pass
        #raise NotImplementedError
        if qpos is None:
            qpos = self.qpos
        is_single = qpos.dim() == 1
        if is_single:
            qpos = qpos[None, :]
        b = qpos.shape[0]
        M = self.M[None, :].expand(b, -1, -1, -1)
        A = self.A[None, :].expand(b, -1, -1)
        Ts = tr.fk_in_space(qpos, M, A)
        if is_single:
            Ts = Ts[0]
        return Ts

    def compute_jacobian(self, qpos=None):
        if qpos is None:
            qpos = self.qpos
        is_single = qpos.dim() == 1
        if is_single:
            qpos = qpos[None, :]
        b = qpos.shape[0]
        M = self.M[None, :].expand(b, -1, -1, -1)
        A = self.A[None, :].expand(b, -1, -1)
        Jac = tr.jacobian_space(qpos, M, A)
        if is_single:
            Jac = Jac[0]
        return Jac

    def inverse_dynamics(self, qpos, qvel, qacc):
        assert qpos.shape == qvel.shape and qvel.shape == qacc.shape
        is_single = qpos.dim() == 1
        if is_single:
            qpos = qpos[None,:]
            qvel = qvel[None,:]
            qacc = qacc[None,:]
        qf = tr.inverse_dynamics(qpos, qvel, qacc, *self.get_parameters(qpos))
        if is_single:
            assert qf.shape == qacc.shape, f"qf: {qf.shape}, qacc: {qacc.shape}, qpos: {qpos.shape}"
            qf = qf[0]
        return qf

    def qacc(self, qpos, qvel, qf):
        is_single = qpos.dim() == 1
        if is_single:
            qpos = qpos[None,:]
            qvel = qvel[None,:]
            qf = qf[None,:]

        qacc = tr.forward_dynamics(qpos, qvel, qf, *self.get_parameters(qpos))
        if is_single:
            qacc = qacc[0]
        #print('qpos, qvel, qacc', qpos, qvel, qacc, qf)
        return qacc

    def step(self):
        # forward ...
        def derivs(state, t):
            # only one batch
            qpos = state[..., :self.dof]
            qvel = state[..., self.dof:self.dof*2]
            qf = state[..., self.dof*2:self.dof*3]
            out = torch.cat((qvel, self.qacc(qpos, qvel, qf), qf*0), dim=-1)
            return out

        state = torch.cat((self.qpos, self.qvel, self.qf), dim=-1)
        # ideally we should do intergral at the euclidea space
        output = tr.rk4(derivs, state, [0, self.timestep])[1]

        self.qpos = (output[..., :self.dof] + np.pi) % (np.pi * 2) - np.pi
        self.qvel = output[..., self.dof:self.dof*2].clamp(-self.max_velocity, self.max_velocity) # velocity is bounded

    def add_link(self, M, screw, type='hinge', range=None):
        """
        :param pose: father pose in 2d
        :param joint_pose: joint_pose
        :param type: type
        :param range: range
        :return:
        """
        assert type == 'hinge'
        assert range is None
        link = Link(M, screw)
        self._links.append(link)
        return link

    def set_ee(self, M):
        self.ee_M = M

    def build(self):
        Mlist = np.array([i.T for i in self._links] + [self.ee_M])
        self.M = torch.tensor(Mlist, dtype=torch.float64, device=self.device)

        # A is the screw in link {i}'s framework
        A = torch.tensor(np.array([i.screw for i in self._links]), dtype=torch.float64, device=self.device)

        self.A = A
        Glist = np.array([i._inertial for i in self._links])
        self.G = torch.tensor(Glist, dtype=torch.float64, device=self.device)

        self.dof = len(self._links)
        self.qpos = self.A.new_zeros((self.dof,)) # theta
        self.qvel = self.A.new_zeros((self.dof,)) # dtheta
        self.qf = self.A.new_zeros((self.dof,))

    def draw_objects(self, viewer):
        qpos = self.qpos
        if qpos.dim() > 1:
            qpos = qpos[0] # only visualize the first environment

        Ts = self.forward_kinematics(qpos)
        for T, link in zip(Ts, self._links):
            # we don't draw the end-effector?
            T = T.detach().cpu().numpy()
            link.draw(viewer, T)

