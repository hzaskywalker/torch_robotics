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
    y,g = net(x)
    input_val = torch.eye(noutputs).reshape(1,noutputs, noutputs).repeat(n, 1, 1).cuda()
    x.retain_grad()
    jac=torch.autograd.grad(y,x,input_val,create_graph=True)
    return y[:,0,:], g[:,0,:], jac[0]


def inverse_model(net_L, q, dq, ddq, return_all=False):
    nbatch=q.shape[0]
    ndim=q.shape[1]
    dqr=dq.unsqueeze(1).repeat(1,ndim,1)
    
    l, g, l_jac=get_batch_jacobian(net_L, q, int(ndim*(ndim+1)/2))
    L = torch.zeros((nbatch, ndim, ndim)).cuda()
    tril_indices = torch.tril_indices(row=ndim, col=ndim, offset=-1)
    
    L[:,tril_indices[0], tril_indices[1]] = l[:,:-ndim]
    L[:,torch.arange(ndim),torch.arange(ndim)]=l[:,-ndim:]
    
    dLdq=torch.zeros((nbatch, ndim, ndim, ndim)).cuda()
    dLdq[:,tril_indices[0], tril_indices[1],:]=l_jac[:,:-ndim]
    dLdq[:,torch.arange(ndim),torch.arange(ndim)]=l_jac[:,-ndim:]   
    dLdt=(dLdq@dqr.unsqueeze(3)).squeeze()
    
    H=L@L.transpose(1,2)
    dHdt=L@dLdt.transpose(1,2)
    dHdt=dHdt+dHdt.transpose(1,2)
    dHdq=dLdq.permute(0,3,1,2)@(L.unsqueeze(1).repeat(1,ndim,1,1).transpose(2,3))
    dHdq=dHdq+dHdq.transpose(2,3)
    
    quad=(dqr.unsqueeze(2)@dHdq@dqr.unsqueeze(3)).squeeze()
    tau=(H@ddq.unsqueeze(2)).squeeze()+(dHdt@dq.unsqueeze(2)).squeeze()-quad/2+g
    if return_all:
        return L,dLdq,dLdt,H,dHdq,dHdt,quad,tau
    else:
        return tau
    
def inverse_model_v2(net_L, q, dq, ddq, return_all=False):
    nbatch=q.shape[0]
    ndim=q.shape[1]
    dqr=dq.unsqueeze(1).repeat(1,ndim,1)
    l, g, l_jac=get_batch_jacobian(net_L, q, int(ndim*(ndim+1)/2))
    L = torch.zeros((nbatch, ndim, ndim)).cuda()
    tril_indices = torch.tril_indices(row=ndim, col=ndim, offset=-1)

    L[:,tril_indices[0], tril_indices[1]] = l[:,:-ndim]
    L[:,torch.arange(ndim),torch.arange(ndim)]=l[:,-ndim:]

    dLdq=torch.zeros((nbatch, ndim, ndim, ndim)).cuda()
    dLdq[:,tril_indices[0], tril_indices[1],:]=l_jac[:,:-ndim]
    dLdq[:,torch.arange(ndim),torch.arange(ndim)]=l_jac[:,-ndim:]

    dLdt=(dLdq@dqr.unsqueeze(3)).squeeze()
    H=L+L.transpose(1,2)
    dHdt=dLdt+dLdt.transpose(-1,-2)
    dHdq=dLdq+dLdq.transpose(-1,-2)
    dHdq=dHdq.permute(0,3,1,2)
    quad=(dqr.unsqueeze(2)@dHdq@dqr.unsqueeze(3)).squeeze()
    tau=(H@ddq.unsqueeze(2)).squeeze()+(dHdt@dq.unsqueeze(2)).squeeze()-quad/2+g
    if return_all:
        return L,dLdq,dLdt,H,dHdq,dHdt,quad,tau
    else:
        return tau
    
class TestModule(nn.Module):
    # Test module only for ndim=3
    def __init__(self):
        super(TestModule, self).__init__()
        self.g=nn.Linear(3,3)
        
    def forward(self, x):
        return torch.cat([torch.exp(x),torch.exp(2*x)],-1), self.g(x)

if __name__ == '__main__':
    model_L=TestModule().cuda()
    model_g=nn.Linear(3,3).cuda()
    q=torch.stack([torch.log(torch.arange(1,4).float()),torch.log(torch.arange(2,5).float()),torch.log(torch.arange(5,8).float())]).cuda()
    dq=torch.ones(q.shape).cuda()
    ddq=torch.ones(q.shape).cuda()

    L,dLdq,dLdt,H,dHdq,dHdt,tau=inverse_model(model_L, model_g, q,dq,ddq, True)
    print(L[0])
    print(dLdq[0])
    print(dLdt[0])
    print(H[0])
    print(dHdq[0])
    print(dHdt[0])
    print(tau[0])