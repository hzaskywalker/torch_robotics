import torch
import numpy as np
from torch import nn
from robot import tr


class ArmModel(nn.Module):
    # possible variants: constrained parameter space
    # possible if we use the dynamics to optimize the geometry
    def __init__(self, dof, dtype=torch.float64, max_velocity=20, action_range=50, timestep=0.1, damping=0.,
                 gravity=(0., -9.8, 0.)):
        super(ArmModel, self).__init__()

        self.dof = dof
        self.max_velocity = max_velocity
        self.action_range = action_range
        self.damping = damping
        self.dtype = dtype
        self.init_parameters(dof, dtype)

        self.gravity = nn.Parameter(torch.tensor(gravity, dtype=dtype), requires_grad=False)
        self.ftip = nn.Parameter(torch.zeros(6, dtype=dtype), requires_grad=False)
        self.timestep = nn.Parameter(torch.tensor(timestep, dtype=dtype), requires_grad=False)

    def init_parameters(self, dof, dtype):
        self._M = nn.Parameter(torch.tensor(np.array([
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] for _ in range(dof+1)]), dtype=dtype),
            requires_grad=True)
        self._A = nn.Parameter(torch.rand((dof, 6), dtype=dtype), requires_grad=True)

        # TODO: perhaps over simplified... we ignored all off-diagonal mass..
        G = [torch.rand((4,), dtype=dtype) for _ in range(dof)] # G should be positive...
        self._G = nn.Parameter(torch.stack(G), requires_grad=True)

    @property
    def G(self):
        out = self._G.new_zeros((*self._G.shape[:-2], self.dof, 6, 6))
        for idx in range(self._G.shape[-2]):
            out[..., idx, :3, :3] = self._G[..., idx, :3].diag()
            out[..., idx, 3, 3] = out[..., idx, 4, 4] = out[..., idx, 5, 5] = self._G[..., idx, 3]
        return out.abs() # project into positive M

    @G.setter
    def G(self, G):
        self._G.data = G[..., torch.arange(4), torch.arange(4)].detach()


    @property
    def A(self):
        # there must be a rotation ...
        A1, A2 = self._A[...,:3], self._A[..., 3:]
        A1 = tr.normalize(A1)
        return torch.cat((A1, A2), dim=-1)

    @A.setter
    def A(self, A):
        self._A.data = A.detach()

    @property
    def M(self):
        # project the parameter into a matrix
        out = torch.zeros_like(self._M)
        #out[:,:3] = self._M[:,:3]
        #out[:,3,:] = 0
        #out[:,3,3] = 1
        #return out

        #rot = self.M_rot
        #out = rot.new_zeros((rot.shape[0], 4, 4))
        a1, a2 = self._M[:, :3, 0], self._M[:, :3, 1]
        assert a1.norm(dim=-1).min() > 1e-15, "check init norm"

        b1 = tr.normalize(a1)
        b2 = tr.normalize(a2 - (a2 * b1).sum(dim=-1, keepdim=True) * b1)
        b3 = torch.cross(b1, b2, dim=-1)
        out[:, :3, 0], out[:,:3,1], out[:, :3, 2] = b1, b2, b3
        out[:, :3, 3] = self._M[:, :3, 3]
        out[:, 3, 3] = 1
        return out

    @M.setter
    def M(self, M):
        self._M.data = M.detach()

    def compute_mass_matrix(self, q):
        return tr.compute_mass_matrix(q, *self.get_parameters(q)[-3:])

    def compute_gravity(self, q):
        params = self.get_parameters(q)
        return tr.compute_passive_force(q, *params[-3:], params[0])

    def compute_coriolis_centripetal(self, q, dq):
        return tr.compute_coriolis_centripetal(q, dq, *self.get_parameters(q)[-3:])

    def inverse_dynamics(self, q, dq, ddq):
        return tr.inverse_dynamics(q, dq, ddq, *self.get_parameters(q))

    def get_parameters(self, qpos, M=None, G=None, A=None):
        b = qpos.shape[0]
        gravity = self.gravity[None,:].expand(b, -1)
        ftip = self.ftip[None,:].expand(b, -1)
        if M is None:
            M = self.M[None,:].expand(b, -1, -1, -1)

        if G is None:
            G = self.G # support batched G
            if G.dim() == 3:
                G = G[None,:].expand(b, -1, -1, -1)

        if A is None:
            A = self.A[None,:].expand(b, -1, -1)

        return gravity, ftip, M, G, A

    def fk(self, q, dim=3):
        params = self.get_parameters(q)
        ee = tr.fk_in_space(q, params[-3], params[-1])
        if dim > 0:
            return ee[:, -1, :dim, 3]
        else:
            return ee[:, -1]

    def qacc(self, qpos, qvel, action, damping=False, **kwargs):
        torque = action * self.action_range
        if damping:
            torque -= self.damping * qvel
        return tr.forward_dynamics(qpos, qvel, torque, *self.get_parameters(qpos))

    def forward(self, state, action, **kwargs):
        # return the
        torque = action * self.action_range

        params = list(self.get_parameters(state))
        M, A = params[-3], params[-1]
        #params[-3], params[-1] = M.detach(), A.detach()

        # variants 2..
        def derivs(state, t):
            # only one batch
            qpos = state[..., :self.dof]
            qvel = state[..., self.dof:self.dof*2]
            qf = state[..., self.dof*2:self.dof*3]
            qf = qf - self.damping * qvel
            qacc = tr.forward_dynamics(qpos, qvel, qf, *params)
            #print(qpos, qvel, qacc, torque)
            out = torch.cat((qvel, qacc, qf*0), dim=-1)
            return out

        state = torch.cat((state, torque), dim=-1)
        new_state = tr.rk4(derivs, state, [0, self.timestep])[1]
        #new_state = state + derivs(state, None) * self.timestep
        new_state = new_state[..., :self.dof * 2]
        q = (new_state[...,:self.dof] + np.pi)%(2*np.pi) - np.pi
        dq = new_state[..., self.dof:].clamp(-self.max_velocity, self.max_velocity)

        ee = tr.fk_in_space(q, M, A)
        return torch.cat((q, dq), -1), ee[:, -1, :2, 3]

    def assign(self, A, M, G):
        self.A = A
        self.M = M
        self.G = G
        return self


def extract_state(obs):
    obs = obs['observation']
    return obs[:4]


def test_model_by_gt():
    from robot import A, U

    env = A.train_utils.make('diff_acrobat')
    mm = env.unwrapped.articulator

    model = ArmModel(2).cuda()
    model.assign(mm)

    obs = env.reset()
    for i in range(50):
        s = extract_state(obs)
        torque = env.action_space.sample()
        obs, _, _, _ = env.step(torque)
        t = extract_state(obs)

        predict = U.tocpu(model( U.togpu(s, torch.float64)[None,:],
                                 U.togpu(torque, torch.float64)[None,:])[0])[0]
        print(((predict - t) ** 2).mean(), predict, t)

def mod(x):
    return (x + np.pi) % (2*np.pi) - np.pi


def test_model_by_training():
    from robot import A, U

    #dataset = A.train_utils.Dataset('/dataset/diff_acrobat')
    #model = ArmModel(2).cuda()
    from robot.model.arm.exp.learn_acrobat_qacc import train
    dataset = A.train_utils.Dataset('/dataset/acrobat2')
    model = ArmModel(2, max_velocity=200, timestep=0.025, damping=0.5).cuda()

    #model.assign()

    #mm = A.train_utils.make('diff_acrobat').unwrapped.articulator
    #model.M.data = mm.M.detach() + torch.randn_like(mm.M)
    #model.A.data = mm.A.detach() + torch.randn_like(mm.A)

    # given the geometrical model, it's easy to estimate the physical parameter
    #model.G.data = mm.G.detach()
    train(model, dataset)


def test_A_grad():
    model = ArmModel(2).cuda()

    inp = torch.zeros((2,), dtype=torch.float64, device='cuda:0')[None,:]
    cc = model.fk(inp)
    cc.mean().backward()
    print(model._A.grad)



if __name__ == '__main__':
    #test_model_by_gt()
    #test_model_by_training()
    test_A_grad()
