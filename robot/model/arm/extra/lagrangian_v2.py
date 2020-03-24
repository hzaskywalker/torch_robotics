import robot
from torch import nn
import torch

# q,dq,ddq: (b x ndim)
# net_L: q(b x ndim)->l(b x ndim*(ndim+1)/2)
# net_g: q(b x ndim)->g(b x ndim)


# q(b x ndim)--net_L-->l(b x ndim*(ndim+1)/2)--reshape-->L(b x ndim x ndim,lower triangle), dLdq(b x ndim x ndim x ndim, lower triangle)
# dLdt(b x ndim x ndim)
# dHdt(b x ndim x ndim)
# dHdq(b x ndim x ndim x ndim)

def get_batch_jacobian(net, x, noutputs):
    x = x.unsqueeze(1)
    n = x.size()[0]
    x = x.repeat(1, noutputs, 1)
    x.requires_grad_(True)
    y = net(x)
    input_val = torch.eye(noutputs).reshape(1,noutputs, noutputs).repeat(n, 1, 1)
    x.retain_grad()
    y.backward(input_val)
    return y[:,0,:], x.grad.data


def inverse_model(net_L, net_g, q, dq, ddq, return_all=False):
    nbatch=q.shape[0]
    ndim=q.shape[1]
    l, l_jac=get_batch_jacobian(net_L, q, int(ndim*(ndim+1)/2))
    L = torch.zeros((nbatch, ndim, ndim))
    tril_indices = torch.tril_indices(row=ndim, col=ndim, offset=0)
    L[:,tril_indices[0], tril_indices[1]] = l
    dLdq=torch.zeros((nbatch, ndim, ndim, ndim))
    dLdq[:,tril_indices[0], tril_indices[1],:]=l_jac
    dLdt=(dLdq@dq.unsqueeze(2)).squeeze()
    H=L@L.transpose(1,2)
    dHdt=L@dLdt.transpose(1,2)+dLdt@L.transpose(1,2)
    dHdq=dLdq.permute(0,3,1,2)@L.transpose(1,2)+L@dLdq.permute(0,3,2,1)
    quad=((dq.unsqueeze(1)@dHdq)@dq.unsqueeze(2)).squeeze() # d(dqHdq)dq
    tau=(H@ddq.unsqueeze(2)).squeeze()+(dHdt@dq.unsqueeze(2)).squeeze()-0.5*quad+net_g(q)
    if return_all:
        return L,dLdq,dLdt,H,dHdq,dHdt,tau
    else:
        return tau

if __name__ == '__main__':
    model_L=TestModule()
    model_g=nn.Linear(3,3)
    q=torch.stack([torch.log(torch.arange(1,4).float()),torch.log(torch.arange(2,5).float()),torch.log(torch.arange(5,8).float())])
    dq=torch.ones(q.shape)
    ddq=torch.ones(q.shape)

    L,dLdq,dLdt,H,dHdq,dHdt,tau=inverse_model(model_L, model_g, q,dq,ddq, True)
    print(L[0])
    print(dLdq[0])
    print(dLdt[0])
    print(H[0])
    print(dHdq[0])
    print(dHdt[0])
    print(tau[0])