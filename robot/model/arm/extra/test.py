from torch import nn
from robot.model.arm.dataset import *

#dataset=Dataset('/dataset/diff_acrobat', device='cuda:0')
#mm = A.train.make('diff_acrobat').unwrapped.articulator
dataset=Dataset('/dataset/acrobat2', device='cuda:0')

from robot.model.arm.exp.phys_model import ArmModel
from robot import tr
class InverseDynamicsModel(ArmModel):
    def __init__(self, dim, dtype):
        super(InverseDynamicsModel, self).__init__(dim, dtype=dtype)

        self.q_fc = nn.Linear(1, 1)
        self.dq_fc = nn.Linear(1, 1)
        self.ddq_fc = nn.Linear(1, 1)
        self.out = nn.Linear(1, 1)

    def forward(self, q, dq, ddq, return_all=False):
        q = (q + np.pi) % (2*np.pi) - np.pi
        q = self.q_fc(q.reshape(-1, 1)).reshape(q.shape)
        dq = self.dq_fc(dq.reshape(-1, 1)).reshape(dq.shape)
        ddq = self.ddq_fc(ddq.reshape(-1, 1)).reshape(ddq.shape)
        tau = tr.inverse_dynamics(q, dq, ddq, *self.get_parameters(q))
        #return [] + [None for i in range(7)]
        #tau = self.out(tau.reshape(-1, 1)).reshape(tau.shape)
        return tau


def get_info(data, ndim):
    q = data[0][:, 1, 0:ndim]
    dq = data[0][:, 1, ndim:2 * ndim]
    ddq = data[0][:, 1, 2 * ndim:3 * ndim]
    tau = data[1].squeeze()

    #dq0 = data[0][:, 0, ndim:2*ndim]
    #ddq0 = data[0][:, 0, 2*ndim:3*ndim]
    #print('real qacc', (dq - dq0)[0], 'estimated qacc', ddq0[0], tau.abs().max())
    #ddq = mm.qacc(q.double(), dq.double(), tau.double() * 50).float()
    #exit(0)
    return q, dq, ddq, tau * 50


class LagrangianNetwork(nn.Module):
    def __init__(self, ndim):
        super(LagrangianNetwork, self).__init__()
        self.feat = nn.Sequential(
            nn.Linear(2 * ndim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.diag = nn.Sequential(
            nn.Linear(256, ndim),
        )
        self.tril = nn.Sequential(
            nn.Linear(256, int(ndim * (ndim - 1) / 2)),
        )
        self.g = nn.Linear(256, ndim)
        self.output = nn.Sequential(nn.Linear(ndim, 256), nn.ReLU(), nn.Linear(256, ndim))

    def get_batch_jacobian(self, q, noutputs):
        n = q.size()[0]
        q.requires_grad_(True)
        q = q.unsqueeze(1)
        q = q.repeat(1, noutputs, 1)
        x = torch.cat((torch.cos(q), torch.sin(q)), dim=-1)
        feature = self.feat(x)
        y = torch.cat([self.tril(feature), self.diag(feature)], dim=-1)
        g = self.g(feature)
        input_val = torch.eye(noutputs).reshape(1, noutputs, noutputs).repeat(n, 1, 1).cuda()
        #     x.retain_grad()
        jac = torch.autograd.grad(y, q, input_val, create_graph=True)
        return y[:, 0, :], g[:, 0, :], jac[0]

    def forward(self, q, dq, ddq, return_all=False):
        nbatch = q.shape[0]
        ndim = q.shape[1]
        dqr = dq.unsqueeze(1).repeat(1, ndim, 1)

        l, g, l_jac = self.get_batch_jacobian(q, int(ndim * (ndim + 1) / 2))
        L = torch.zeros((nbatch, ndim, ndim)).cuda()
        tril_indices = torch.tril_indices(row=ndim, col=ndim, offset=-1)

        L[:, tril_indices[0], tril_indices[1]] = l[:, :-ndim]
        L[:, torch.arange(ndim), torch.arange(ndim)] = l[:, -ndim:]

        dLdq = torch.zeros((nbatch, ndim, ndim, ndim)).cuda()
        dLdq[:, tril_indices[0], tril_indices[1], :] = l_jac[:, :-ndim]
        dLdq[:, torch.arange(ndim), torch.arange(ndim)] = l_jac[:, -ndim:]
        dLdt = (dLdq @ dqr.unsqueeze(3)).squeeze()

        H = L @ L.transpose(1, 2)
        dHdt = L @ dLdt.transpose(1, 2)
        dHdt = dHdt + dHdt.transpose(1, 2)
        dHdq = dLdq.permute(0, 3, 1, 2) @ (L.unsqueeze(1).repeat(1, ndim, 1, 1).transpose(2, 3))
        dHdq = dHdq + dHdq.transpose(2, 3)

        quad = (dqr.unsqueeze(2) @ dHdq @ dqr.unsqueeze(3)).squeeze()
        tau = (H @ ddq.unsqueeze(2)).squeeze() + (dHdt @ dq.unsqueeze(2)).squeeze() - quad / 2 + g
        #tau = self.output(tau)
        if return_all:
            return L, dLdq, dLdt, H, dHdq, dHdt, quad, tau
        else:
            return tau

torch.cuda.set_device('cuda:0')
# ndim=7 for arm, ndim=2 for acrobat
ndim = 2

#model_lag = LagrangianNetwork(ndim)
model_lag = InverseDynamicsModel(2, dtype=torch.float)
model_lag = model_lag.cuda()

model_naive = nn.Sequential(
    nn.Linear(3 * ndim, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, ndim),
)

model_naive = model_naive.cuda()
crit = nn.MSELoss()

train_loss_lag = []
train_loss_naive = []
val_loss_lag = []
val_loss_naive = []

optimizer_lag = torch.optim.Adam(model_lag.parameters(), lr=0.1)
optimizer_naive = torch.optim.Adam(model_naive.parameters(), lr=1e-4)

for t in range(5):
    for i in range(dataset.num_train):
        data = dataset.sample()
        q, dq, ddq, tau_target = get_info(data, ndim)
        #         tau_target=tau_target/50

        # Lagrangian network
        tau_lag = model_lag(q, dq, ddq)
        optimizer_lag.zero_grad()
        loss_lag = crit(tau_lag, tau_target)
        loss_lag_rel = torch.mean((tau_lag - tau_target) ** 2 / (tau_target) ** 2)
        loss_lag.backward()
        optimizer_lag.step()

        # Naive network
        tau = model_naive(torch.cat([q, dq, ddq], axis=-1))
        optimizer_naive.zero_grad()
        loss_naive = crit(tau, tau_target)
        loss_naive_rel = torch.mean((tau - tau_target) ** 2 / (tau_target) ** 2)
        loss_naive.backward()
        optimizer_naive.step()

        if i % 1000 == 0:
            print(tau_lag[0], tau[0], tau_target[0])
            print('mse lag', crit(tau_lag[0], tau_target[0]))
            print('mse naive', crit(tau[0], tau_target[0]))
            print('relative lag', torch.mean((tau_lag[0] - tau_target[0]) ** 2 / (tau_target[0]) ** 2))
            print('relative naive', torch.mean((tau[0] - tau_target[0]) ** 2 / (tau_target[0]) ** 2))

            print(i, 'MSE:', loss_lag.data.item(), loss_naive.data.item(), 'Relative MSE:', loss_lag_rel.data.item(),
                  loss_naive_rel.data.item(), )
            vdata = dataset.sample('valid')
            q, dq, ddq, tau_target = get_info(vdata, ndim)

            # Lagrangian network
            #L_lag, dLdq_lag, dLdt_lag, H_lag, dHdq_lag, dHdt_lag, quad_lag, tau = model_lag(q, dq, ddq, True)
            tau = model_lag(q, dq, ddq, True)

            loss_lag = crit(tau, tau_target)
            loss_lag_rel = torch.mean((tau - tau_target) ** 2 / (tau_target) ** 2)
            #             train_loss_lag.append(loss_lag.data.item())
            #             val_loss_lag.append(loss_val_lag.item())

            #             # Naive network
            tau = model_naive(torch.cat([q, dq, ddq], axis=1))
            loss_naive = crit(tau, tau_target)
            loss_naive_rel = torch.mean((tau - tau_target) ** 2 / (tau_target) ** 2)
            #             train_loss_naive.append(loss_naive.data.item())
            #             val_loss_naive.append(loss_val_naive.item())

            print('VAL', i, 'MSE:', loss_lag.data.item(), loss_naive.data.item(), 'Relative MSE:',
                  loss_lag_rel.data.item(), loss_naive_rel.data.item(), )
