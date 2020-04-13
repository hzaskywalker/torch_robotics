import torch
import numpy as np
from torch import nn
from robot import tr


class AngleLoss(nn.Module):
    def forward(self, predict, label):
        diff = torch.abs(predict - label)
        return (torch.min(diff, 2 * np.pi - diff) ** 2).mean()

class ArmModel(nn.Module):
    # possible variants: constrained parameter space
    # possible if we use the dynamics to optimize the geometry
    def __init__(self, dof, dtype=torch.float64, max_velocity=20, action_range=50, timestep=0.1):
        super(ArmModel, self).__init__()

        self.dof = dof
        self.max_velocity = max_velocity
        self.action_range = action_range
        self._M = nn.Parameter(torch.tensor(np.array([
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] for _ in range(dof+1)]), dtype=dtype),
                              requires_grad=True)

        #self.M_rot = nn.Parameter(torch.tensor(np.array([
        #    [[1, 0], [0, 1], [0, 0]] for _ in range(dof+1)]), dtype=dtype),
        #                      requires_grad=True)
        #self.M_trans = nn.Parameter(torch.tensor(np.array([[0, 0, 0] for _ in range(dof+1)]), dtype=dtype),
        #                      requires_grad=True)

        # NOTE: A should be greater than 0
        self.A = nn.Parameter(torch.rand((dof, 6), dtype=dtype), requires_grad=True)

        #G = [torch.rand((6), dtype=dtype).diag() for _ in range(dof)] # G should be positive...

        # not feasible, but we ignore the off-diagonal elements

        # TODO: perhaps over simplified... we ignored all off-diagonal mass..
        G = [torch.rand((4,), dtype=dtype) for _ in range(dof)] # G should be positive...
        self._G = nn.Parameter(torch.stack(G), requires_grad=True)

        #L = [torch.randn((6, 6), dtype=dtype) for _ in range(dof)]
        #self.L = nn.Parameter(torch.stack(L), requires_grad=True)

        self.gravity = nn.Parameter(torch.tensor([0., -9.8, 0.], dtype=dtype), requires_grad=False)
        self.ftip = nn.Parameter(torch.zeros(6, dtype=dtype), requires_grad=False)
        self.timestep = nn.Parameter(torch.tensor(timestep, dtype=dtype), requires_grad=False)

    @property
    def G(self):
        out = self._G.new_zeros((self.dof, 6, 6))
        for idx, i in enumerate(self._G):
            out[idx, :3,:3] = i[:3].diag()
            out[idx, 3,3] = out[idx, 4, 4] = out[idx, 5, 5] = i[3]
        return out

    """
    @property
    def G(self):
        return tr.dot(self.L, self.L.transpose(-1, -2))
    """
    def compute_mass_matrix(self, q):
        return tr.compute_mass_matrix(q, *self.get_parameters(q)[-3:])

    def compute_gravity(self, q):
        params = self.get_parameters(q)
        return tr.compute_passive_force(q, *params[-3:], params[0])

    def compute_coriolis_centripetal(self, q, dq):
        return tr.compute_coriolis_centripetal(q, dq, *self.get_parameters(q)[-3:])

    @property
    def M(self):
        out = torch.zeros_like(self._M)
        out[:,:3] = self._M[:,:3]
        out[:,3,:] = 0
        out[:,3,3] = 1
        return out

        rot = self.M_rot
        out = rot.new_zeros((rot.shape[0], 4, 4))
        a1, a2 = rot[:,:,0], rot[:, :, 1]

        b1 = tr.normalize(a1)
        b2 = tr.normalize(a2 - (a2 * b1).sum(dim=-1, keepdim=True) * b1)
        b3 = torch.cross(b1, b2, dim=-1)
        out[:, :3, 0], out[:,:3,1], out[:, :3, 2] = b1, b2, b3
        out[:, :3, 3] = self.M_trans
        out[:, 3, 3] = 1
        return out

    def get_parameters(self, qpos):
        b = qpos.shape[0]
        gravity = self.gravity[None,:].expand(b, -1)
        ftip = self.ftip[None,:].expand(b, -1)
        M = self.M[None,:].expand(b, -1, -1, -1)
        G = self.G[None,:].expand(b, -1, -1, -1)
        A = self.A[None,:].expand(b, -1, -1)
        return gravity, ftip, M, G, A

    def fk(self, q):
        params = self.get_parameters(q)
        print(params[-1].shape)
        ee = tr.fk_in_space(q, params[-3], params[-1])
        return ee[:, -1, :3, 3]

    def qacc(self, qpos, qvel, action):
        torque = action * self.action_range
        return tr.forward_dynamics(qpos, qvel, torque, *self.get_parameters(qpos))

    def forward(self, state, torque):
        # return the
        torque *= self.action_range

        params = list(self.get_parameters(state))
        M, A = params[-3], params[-1]
        #params[-3], params[-1] = M.detach(), A.detach()

        # variants 2..
        def derivs(state, t):
            # only one batch
            qpos = state[..., :self.dof]
            qvel = state[..., self.dof:self.dof*2]
            qf = state[..., self.dof*2:self.dof*3]
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
        return torch.cat((q, dq), -1), ee[:, -1, :3, 3]

    def assign(self, mm):
        #self.M.data = mm.M.detach()
        self.A.data = mm.A.detach()
        self.M_rot.data = mm.M[:,:3,:2]
        self.M_trans.data = mm.M[:,:3,3]
        #self.G.data = mm.G.detach()
        self.L.data = torch.stack([torch.cholesky(g) for g in mm.G.detach()])
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

def train(model, dataset):
    #model.assign(mm)
    from robot import U

    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    loss_fn2 = AngleLoss()

    def eval_predict(output, t):
        predict, ee = output
        dq_loss = loss_fn(predict[..., 2:], t[..., 2:4])
        q_loss = loss_fn2(predict[..., :2], t[..., :2])
        ee_loss = loss_fn(ee[..., :2], t[..., -2:]) + (ee[...,2] **2).mean()
        #print('q', q_loss, 'ee', ee_loss, 'dq', dq_loss)
        return dq_loss + q_loss + ee_loss

    def validate(num_iter=20):
        model.eval()
        total = 0
        for i in range(num_iter):
            data = dataset.sample('valid', 256, timestep=2)
            s = data[0][:, 0, :4].double()
            t = data[0][:, 1].double()

            a = data[1][:, 0].double()
            print(data[0][0, 0, -2:])
            total += eval_predict(model(s[:, :], a), t)  # action_range = 50
        model.train()
        return U.tocpu(total/num_iter)

    import tqdm
    for i in tqdm.trange(10000):
        data = dataset.sample('train', 256, timestep=2)
        s = data[0][:, 0, :4].double()
        t = data[0][:, 1].double()
        a = data[1][:, 0].double()

        optim.zero_grad()
        predict, ee = model(s, a)
        loss = eval_predict((predict, ee), t)  # action_range = 50
        loss.backward()
        optim.step()

        if i % 100 == 0:
            print("learned G:", model.G)
            print("learned M:", model.M)
            print("learned A:", model.A)
            #print("env G:", mm.G[1])
            print('mse loss', U.tocpu(loss), predict[0], t[0])
            print('valid mse loss', validate(5))

def test_model_by_training():
    from robot import A, U

    dataset = A.train_utils.Dataset('/dataset/diff_acrobat')
    model = ArmModel(2).cuda()
    #dataset = A.train_utils.Dataset('/dataset/acrobat2')
    #model = ArmModel(2, max_velocity=100, timestep=0.025).cuda()

    #model.assign()

    #mm = A.train_utils.make('diff_acrobat').unwrapped.articulator
    #model.M.data = mm.M.detach() + torch.randn_like(mm.M)
    #model.A.data = mm.A.detach() + torch.randn_like(mm.A)

    # given the geometrical model, it's easy to estimate the physical parameter
    #model.G.data = mm.G.detach()
    train(model, dataset)



if __name__ == '__main__':
    #test_model_by_gt()
    test_model_by_training()
