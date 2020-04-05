import torch
import numpy as np
from torch import nn
from robot import tr


class AngleLoss(nn.Module):
    def forward(self, predict, label):
        diff = torch.abs(predict - label)
        return (torch.min(diff, 2 * np.pi - diff) ** 2).mean()

class ArmModel(nn.Module):
    def __init__(self, dof, dtype=torch.float64, max_velocity=20):
        super(ArmModel, self).__init__()

        self.dof = dof
        self.max_velocity = max_velocity

        initM = [[1, 0, 0, 0],
            [0, 1, 0, 0.],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]


        self.M = nn.Parameter(torch.tensor(np.array([initM for _ in range(dof+1)]), dtype=dtype),
                              requires_grad=True)

        # NOTE: A should be greater than 0
        self.A = nn.Parameter(torch.rand((dof, 6), dtype=dtype), requires_grad=True)

        G = [torch.rand((6), dtype=dtype).diag() for _ in range(dof)] # G should be positive...
        self.G = nn.Parameter(torch.stack(G), requires_grad=True)

        self.gravity = nn.Parameter(torch.tensor([0., -9.8, 0.], dtype=dtype), requires_grad=False)
        self.ftip = nn.Parameter(torch.zeros(6, dtype=dtype), requires_grad=False)
        self.timestep = nn.Parameter(torch.tensor(0.1, dtype=dtype), requires_grad=False)

    def get_parameters(self, qpos):
        b = qpos.shape[0]
        gravity = self.gravity[None,:].expand(b, -1)
        ftip = self.ftip[None,:].expand(b, -1)
        M = self.M[None,:].expand(b, -1, -1, -1)
        G = self.G[None,:].expand(b, -1, -1, -1)
        A = self.A[None,:].expand(b, -1, -1)
        return gravity, ftip, M, G, A

    def qacc(self, qpos, qvel, qf):
        is_single = qpos.dim() == 1
        if is_single:
            qpos = qpos[None,:]
            qvel = qvel[None,:]
            qf = qf[None,:]

        qacc = tr.forward_dynamics(qpos, qvel, qf, *self.get_parameters(qpos))
        if is_single:
            qacc = qacc[0]
        return qacc

    def forward(self, state, torque):
        # return the
        def derivs(state, t):
            # only one batch
            qpos = state[..., :self.dof]
            qvel = state[..., self.dof:self.dof*2]
            qf = state[..., self.dof*2:self.dof*3]
            out = torch.cat((qvel, self.qacc(qpos, qvel, qf), qf*0), dim=-1)
            return out

        state = torch.cat((state, torque), dim=-1)
        new_state = tr.rk4(derivs, state, [0, self.timestep])[1][...,:self.dof * 2]
        q = (new_state[...,:self.dof] + np.pi)%(2*np.pi) - np.pi
        dq = new_state[..., self.dof:].clamp(-self.max_velocity, self.max_velocity)
        return torch.cat((q, dq), -1)

    def assign(self, mm):
        self.M.data = mm.M.detach()
        self.A.data = mm.A.detach()
        self.G.data = mm.G.detach()
        return self


def extract_state(obs):
    obs = obs['observation']
    return obs[:4]


def test_model_by_gt():
    from robot import A, U

    env = A.train.make('diff_acrobat')
    mm = env.unwrapped.articulator

    model = ArmModel(2).cuda()
    model.assign(mm)

    action_range = env.unwrapped.action_range

    obs = env.reset()
    for i in range(50):
        s = extract_state(obs)
        torque = env.action_space.sample()
        obs, _, _, _ = env.step(torque)
        t = extract_state(obs)

        predict = U.tocpu(model( U.togpu(s, torch.float64)[None,:],
                                 U.togpu(torque * action_range, torch.float64)[None,:])[0])
        print(((predict - t) ** 2).mean(), predict, t)


def test_model_by_training():
    from robot import A, U

    dataset = A.train.Dataset('/dataset/diff_acrobat')
    model = ArmModel(2).cuda()

    optim = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    loss_fn2 = AngleLoss()

    #model.assign()

    mm = A.train.make('diff_acrobat').unwrapped.articulator
    #model.M.data = mm.M.detach() + torch.randn_like(mm.M)
    #model.A.data = mm.A.detach() + torch.randn_like(mm.A)

    # given the geometrical model, it's easy to estimate the physical parameter
    #model.G.data = mm.G.detach()

    def validate(num_iter=20):
        model.eval()
        total = 0
        for i in range(num_iter):
            data = dataset.sample('valid', 256, timestep=2)
            s = data[0][:, 0, :4].double()
            t = data[0][:, 1, :4].double()
            a = data[1][:, 0].double()
            predict = model(s[:, :], a * 50)  # action_range = 50
            loss = loss_fn(predict[..., 2:], t[..., 2:]) + loss_fn2(predict[..., :2], t[..., :2])
            total += loss
        model.train()
        return U.tocpu(total/num_iter)

    import tqdm
    for i in tqdm.trange(10000):
        data = dataset.sample('train', 256, timestep=2)
        s = data[0][:, 0, :4].double()
        t = data[0][:, 1, :4].double()
        a = data[1][:, 0].double()

        optim.zero_grad()
        predict = model(s[:,:], a * 50)  #action_range = 50
        loss = loss_fn(predict[...,2:], t[...,2:]) + loss_fn2(predict[...,:2], t[...,:2])
        loss.backward()
        optim.step()

        if i % 100 == 0:
            print("learned G:", model.G[1])
            print("env G:", mm.G[1])
            print('mse loss', U.tocpu(loss), predict[0], t[0])
            print('valid mse loss', validate(5))


if __name__:
    #test_model_by_gt()
    test_model_by_training()