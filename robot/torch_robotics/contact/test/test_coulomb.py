# This one is tooooo complex...
# I have to check it separately ...
import numpy as np
from robot import tr
from robot.torch_robotics.contact.elastic import coulomb_friction

def gen():
    n_c = 3
    contact_dof = 6

    n_varaible = n_c * contact_dof

    L = np.random.normal(size=(n_varaible, n_varaible))
    M = L.T.dot(L)
    A = np.linalg.inv(M)
    a0 = np.random.normal(size=(n_varaible,))
    v0 = np.random.normal(size=(n_varaible,))
    d0 = np.random.normal(size=(n_c,)) * 0
    return A, a0, v0, d0

def get_f(f1, beta):
    out = []
    beta = beta.reshape((10, f1.shape[0]))
    for i in range(f1.shape[0]):
        out.append(np.concatenate(((f1[i],), beta[:5, i]-beta[5:, i])))
    return np.array(out).T.reshape(-1)

def get_v1(A, a0, v0, f, h):
    return h*(A@f+a0)+v0

def get_d1(v1, d0, h, alpha0):
    # this is the constraints on the normal direction
    v1 = v1.reshape(6, -1)[0]
    return v1 * h + d0 - alpha0

def get_beta_complementarity(v, lamb):
    v = v.reshape(6, -1)
    DTv = np.concatenate((v[1:], -v[1:]), axis=0)
    return (lamb[None, :] + DTv).reshape(-1)

def get_lambda_complementarity(f1, beta, mu):
    #return mu * f1 - beta.sum()
    return mu * f1 - beta.reshape(10, -1).sum(axis=0)

def test():
    A, a0, v0, d0 = gen()
    alpha0 = 0.1
    mu = 0.5
    h = 0.1

    A_gpu = tr.togpu(A)[None,:]
    a0_gpu = tr.togpu(a0)[None,:]
    v0_gpu = tr.togpu(v0)[None,:]
    d0_gpu = tr.togpu(d0)[None,:]

    contact_dof = 6
    nc = 3
    X, Y, VX, VY, F = coulomb_friction(contact_dof, A_gpu, a0_gpu, v0_gpu, d0_gpu, alpha0, mu, h)

    f1 = np.random.normal(size=(nc,))
    beta = np.random.normal(size=(nc*(contact_dof * 2 - 2),))
    lamb = np.random.normal(size=(nc,))
    x = np.concatenate((f1,beta, lamb) )

    f = get_f(f1, beta)
    print(f-F.detach().cpu().numpy()@x)

    v1 = get_v1(A, a0, v0, f, h)

    print(v1-VX.detach().cpu().numpy()@x-VY.detach().cpu().numpy())

    y = np.concatenate((get_d1(v1, d0, h, alpha0),
                        get_beta_complementarity(v1, lamb),
                        get_lambda_complementarity(f1, beta, mu)))

    #print(X[0][-1])
    #print(beta[-10:].sum())
    y2 = X[0].detach().cpu().numpy()@x+Y[0].detach().cpu().numpy()
    print(y-y2)
    #print(beta.reshape(10, -1).sum(axis=0))



if __name__ == '__main__':
    test()
