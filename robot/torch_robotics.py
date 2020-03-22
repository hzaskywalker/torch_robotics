# torch robotics library
# simulate a robot arm with pytorch
# this is mostly based on the modern_robotics.py

import torch
import numpy as np


def trans_to_Rp(T):
    """Converts a homogeneous transformation matrix into a rotation matrix
    and position vector
    :param T: A homogeneous transformation matrix
    :return R: The corresponding rotation matrix,
    :return p: The corresponding position vector.
    """
    return T[..., 0: 3, 0: 3], T[..., 0: 3, 3]


def transpose(R):
    dim = len(R.shape)
    Rt = R.permute(*range(dim-2), dim-1, dim-2) # (b, 3, 3)
    return Rt


def dot(A, B):
    if A.dim() == 3 and B.dim() == 3:
        return A @ B
    elif A.dim() ==3 and B.dim() == 2:
        return (A @ B[..., None])[..., 0]
    else:
        raise NotImplementedError


def normalize(V):
    """Normalizes a vector
    :param V: A vector
    :return: A unit vector pointing in the same direction as z
    """
    return V / torch.norm(V, dim=-1)


def NearZero(z):
    """Determines whether a scalar is small enough to be treated as zero
    :param z: A scalar input to check
    :return: True if z is close to zero, false otherwise
    """
    return z.abs() < 1e-6


def make_matrix(arrays):
    # all elements of arrays has the same shape
    return torch.cat([torch.cat(i, dim=-1) for i in arrays], dim=-2)


def vec_to_so3(omg):
    """Converts a 3-vector to an so(3) representation
    :param omg: A 3-vector
    :return: The skew symmetric representation of omg
    """
    omg0, omg1, omg2 = omg[...,0], omg[..., 1], omg[..., 2]
    ans = torch.zeros(omg.shape[:-1]+(3,3), device=omg.device, dtype=omg.dtype)
    ans[..., 0, 1] = -omg2
    ans[..., 0, 2] = omg1
    ans[..., 1, 0] = omg2
    ans[..., 1, 2] = -omg0
    ans[..., 2, 0] = -omg1
    ans[..., 2, 1] = omg0
    return ans


def so3_to_vec(so3mat):
    """Converts an so(3) representation to a 3-vector
    :param so3mat: A 3x3 skew-symmetric matrix
    :return: The 3-vector corresponding to so3mat
    """
    #return np.array([so3mat[2][1], so3mat[0][2], so3mat[1][0]])
    return torch.stack((so3mat[..., 2, 1], so3mat[..., 0, 2], so3mat[..., 1, 0]), dim=-1)


def vec_to_se3(V):
    """Converts a spatial velocity vector into a 4x4 matrix in se3
    :param V: A 6-vector representing a spatial velocity
    :return: The 4x4 se3 representation of V
    """
    #return np.r_[np.c_[VecToso3([V[0], V[1], V[2]]), [V[3], V[4], V[5]]],
    #             np.zeros((1, 4))]
    return make_matrix((
        (vec_to_so3(V[..., :3]), V[..., 3:, None]),
        (V.new_zeros((*V.shape[:-1], 1, 4)), )
    ))


def safe_div(a, b):
    b += (torch.abs(b) < 1e-15).float() * 1e-14 # make sure it's not zero
    return a / b


def eyes_like(mat, n=None):
    if n is None:
        n = mat.shape[-1]
    eye = torch.eye(n, device=mat.device, dtype=mat.dtype)
    while len(eye.shape) < len(mat.shape):
        eye = eye[None,:]
    return eye.expand(*mat.shape[:-2], -1, -1)


def expso3(so3mat):
    """Computes the matrix exponential of a matrix in so(3)
    :param so3mat: A 3x3 skew-symmetric matrix
    :return: The matrix exponential of so3mat
    """
    omgtheta = so3_to_vec(so3mat) # (b, 3)
    mask = NearZero(torch.norm(omgtheta, dim=-1))

    eye = eyes_like(so3mat, 3)
    theta = torch.norm(omgtheta, dim=-1) #(b,)
    omgmat = safe_div(so3mat, theta[..., None, None])

    theta = theta[..., None, None]
    Rodrigue =  eye + torch.sin(theta) * omgmat \
            + (1-torch.cos(theta)) * dot(omgmat, omgmat)

    mask = mask[..., None, None].float()
    return mask * eye + (1-mask) * Rodrigue # if near zero, just return identity...


def Rp_to_trans(R, p):
    a = R.new_zeros((*R.shape[:-2], 4, 4))
    a[..., :3,:3] = R
    a[..., :3,3] = p
    a[..., -1, -1] = 1
    return a


def expse3(se3mat):
    """Computes the matrix exponential of an se3 representation of
    exponential coordinates
    :param se3mat: A matrix in se3
    :return: The matrix exponential of se3mat

    page 105 on the book
    """
    a, b, c = se3mat[..., 2, 1], se3mat[..., 0, 2], se3mat[..., 1, 0]
    theta = (a**2+b**2+c**2)**0.5
    is_translation = NearZero(theta).float()

    eye = eyes_like(se3mat, 3)

    safe_theta = theta + is_translation * 1e-14 # make sure it's not zero

    #omgmat = safe_div(se3mat[...,:3,:3], theta[..., None, None])
    omgmat = se3mat[..., :3, :3]/safe_theta[..., None, None]
    v = se3mat[..., :3, 3]/safe_theta[..., None]

    theta = theta[..., None, None]
    omgmat2 = dot(omgmat, omgmat)
    acos = 1-torch.cos(theta)
    si = torch.sin(theta)

    R2 =  (si * omgmat + acos * omgmat2) * (1-is_translation[..., None, None])
    R2 += eye
    tmp = eye * theta + acos * omgmat + (theta - si) * omgmat2
    p2 = dot(tmp, v)

    p2 = is_translation[..., None] * se3mat[..., 0:3, 3] + (1-is_translation)[..., None] * p2
    return Rp_to_trans(R2, p2)


def inv_trans(T):
    """Inverts a homogeneous transformation matrix
    :param T: A homogeneous transformation matrix
    :return: The inverse of T
    Uses the structure of transformation matrices to avoid taking a matrix
    inverse, for efficiency.
    """
    R, p = trans_to_Rp(T)
    Rt = transpose(R)
    p2 = -dot(Rt, p)
    return Rp_to_trans(Rt, p2)


def Adjoint(T):
    """Computes the adjoint representation of a homogeneous transformation
    matrix
    :param T: A homogeneous transformation matrix
    :return: The 6x6 adjoint representation [AdT] of T
    """
    R, p = trans_to_Rp(T)
    zero = torch.zeros_like(R)
    #return make_matrix((
    #    (R, zero),
    #    (dot(vec_to_so3(p), R), R)
    #))
    zero = R.new_zeros((*R.shape[:-2], 6, 6))
    zero[...,:3,:3] = R
    zero[...,3:,:3] = dot(vec_to_so3(p), R)
    zero[...,3:,3:] = R
    return zero


def fk_in_space(theta, M, A):
    """
    :param theta: qpos
    :param M: home transformation
    :param S: screw axis in space frame and the home configuration
    :return:
    """
    n = theta.shape[1]
    #Mi = eyes_like(M[..., 0, :, :], 4)
    T = eyes_like(M[..., 0, :, :], 4)
    outputs = []
    for i in range(n):
        #Mi = dot(Mi, M[:, i])
        #Ai = dot(Adjoint(inv_trans(Mi)), S[:, i])
        Ai = A[:, i]
        Ti = dot(M[:, i], expse3(vec_to_se3(Ai * theta[:, i, None])))
        T = dot(T, Ti)
        outputs.append(T)
    outputs.append(dot(T, M[:, n]))
    return torch.stack(outputs, dim=1)


def ad(V):
    """Calculate the 6x6 matrix [adV] of the given 6-vector
    :param V: A 6-vector spatial velocity
    :return: The corresponding 6x6 matrix [adV]
    Used to calculate the Lie bracket [V1, V2] = [adV1]V2
    """
    #omgmat = VecToso3([V[0], V[1], V[2]])
    assert V.shape[-1] == 6
    omgmat = vec_to_so3(V[..., :3])
    vmat = vec_to_so3(V[..., 3:])
    adv = omgmat.new_zeros(V.shape[:-1]+(6, 6))
    adv[..., :3,:3] = omgmat
    adv[..., 3:,3:] = omgmat
    adv[..., 3:,:3] = vmat
    return adv


def newton_law(G, V, dV):
    return dot(G, dV) - dot(dot(transpose(ad(V)), G), V)

def S_to_A(S, M):
    Mi = eyes_like(M[:, 0, :, :], 4) #
    A = []
    for i in range(S.shape[1]):
        Mi = dot(Mi, M[:, i])
        Ai = dot(Adjoint(inv_trans(Mi)), S[:, i])
        A.append(Ai)
    return torch.stack(A, dim=1)


def inverse_dynamics(theta, dtheta, ddtheta, gravity, Ftip, M, G, A):
    """Computes inverse dynamics in the space frame for an open chain robot
    :param thetalist: n-vector of joint variables
    :param dthetalist: n-vector of joint rates
    :param ddthetalist: n-vector of joint accelerations
    :param g: Gravity vector g
    :param Ftip: Spatial force applied by the end-effector expressed in frame
                 {n+1}
    :param Mlist: List of link frames {i} relative to {i-1} at the home
                  position, matrix of (b, n+1, 4, 4)
    :param Glist: Spatial inertia matrices Gi of the links, matrix of (b, n, 6, 6)
    :param Alist: Screw axes Si of the joints in frame {i}, matrix of (b, n, 6)
    :return: The n-vector of required joint forces/torques
    This function uses forward-backward Newton-Euler iterations to solve the
    equation:
    taulist = Mlist(thetalist)ddthetalist + c(thetalist,dthetalist) \
              + g(thetalist) + Jtr(thetalist)Ftip
    """
    batch_shape = theta.shape[:-1]
    assert len(batch_shape) == 1

    n = theta.shape[-1]
    #Mi = torch.eye(4)

    A_flat = A.reshape(-1, *A.shape[2:]) # on batch version
    theta_flat = theta.reshape(-1)[:, None]
    T = dot(expse3(vec_to_se3(A_flat * -theta_flat)), inv_trans(M[:, :n].reshape(-1, 4, 4)))
    AdT = Adjoint(T).view(-1, n, 6, 6)

    Vi = theta.new_zeros(batch_shape + (6,)) # initial Vi
    V = []
    for i in range(n):
        Vi = A[:, i] * dtheta[:, i, None] + dot(AdT[:, i], Vi)
        V.append(Vi) # new Vi
    V = torch.stack(V, dim=1)
    V_flat = V.view(-1, 6)

    dV_flat = dot(ad(V_flat), A_flat) * dtheta.reshape(-1)[:, None] + A_flat * ddtheta.reshape(-1)[:, None]
    dV_flat = dV_flat.view(-1, n, 6)

    dV = []
    dVi = theta.new_zeros(batch_shape + (6,)) # initial Vi
    dVi[:, -3:] = -gravity # we need the anti velocity to overcome gravity
    for i in range(n):
        dVi= dot(AdT[:, i], dVi) + dV_flat[:, i]
        dV.append(dVi)
    dV = torch.stack(dV, dim=1)

    dV_flat = dV.view(-1, 6)
    G_flat = G.reshape(-1, 6, 6)
    total_F = newton_law(G_flat, V_flat, dV_flat).view(-1, n, 6)

    Fi = Ftip
    last_AdT = Adjoint(inv_trans(M[:, n]))
    tau = []
    for i in range(n-1, -1, -1):
        Fi = total_F[:, i] + dot(transpose(last_AdT), Fi)
        tau.append((Fi * A[:, i]).sum(dim=-1))
        last_AdT = AdT[:, i]

    return torch.stack(tau[::-1], dim=-1)


def compute_mass_matrix(theta, M, G, A):
    """Computes the mass matrix of an open chain robot based on the given
    configuration"""
    def expand_n(array, n):
        return array[None, :].expand(n, *((-1,) * array.dim())).reshape(-1, *array.shape[1:])

    assert theta.dim() == 2

    n = theta.shape[-1]
    ddtheta = eyes_like(M[..., 0, :, :], n=n).permute(1, 0, 2).reshape(-1, n)

    theta = expand_n(theta, n)
    M = expand_n(M, n)
    G = expand_n(G, n)
    A = expand_n(A, n)
    gravity = theta.new_zeros((theta.shape[0], 3))
    ftip = theta.new_zeros((theta.shape[0], 6))

    M = inverse_dynamics(theta, theta * 0, ddtheta, gravity, ftip, M, G, A)
    M = M.reshape(n, -1, n).permute(1, 0, 2).contiguous()
    return M


def compute_coriolis_centripetal(theta, dtheta, M, G, A):
    """
    compute the force, but not the matrix... of course I can do something to retrive the matrix just like the mass matrix.
    """
    gravity = theta.new_zeros((theta.shape[0], 3))
    ftip = theta.new_zeros((theta.shape[0], 6))
    return inverse_dynamics(theta, dtheta, dtheta * 0, gravity, ftip, M, G, A)


def compute_passive_force(theta, M, G, A, gravity=None, ftip=None):
    zero_gravity = theta.new_zeros((theta.shape[0], 3))
    zero_ftip = theta.new_zeros((theta.shape[0], 6))

    g, f = None, None
    if gravity is not None:
        g = inverse_dynamics(theta, theta * 0, theta * 0, gravity, zero_ftip, M, G, A)
    if ftip is not None:
        f = inverse_dynamics(theta, theta * 0, theta * 0, zero_gravity, ftip, M, G, A)
    return g, f


def forward_dynamics(theta, dtheta, tau, gravity, Ftip, M, G, A):
    mass = compute_mass_matrix(theta, M, G, A)
    c = compute_coriolis_centripetal(theta, dtheta, M, G, A)
    g, f = compute_passive_force(theta, M, G, A, gravity, Ftip)
    #TODO: delete this
    return dot(torch.inverse(mass), tau-c-g-f)



def rk4(derivs, y0, t, *args, **kwargs):
    """
    Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
    This is a toy implementation which may be useful if you find
    yourself stranded on a system w/o scipy.  Otherwise use
    :func:`scipy.integrate`.
    *y0*
        initial state vector
    *t*
        sample times
    *derivs*
        returns the derivative of the system and has the
        signature ``dy = derivs(yi, ti)``
    *args*
        additional arguments passed to the derivative function
    *kwargs*
        additional keyword arguments passed to the derivative function
    Example 1 ::
        ## 2D system
        def derivs6(x,t):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)
    Example 2::
        ## 1D system
        alpha = 2
        def derivs(x,t):
            return -alpha*x + exp(-t)
        y0 = 1
        yout = rk4(derivs, y0, t)
    If you have access to scipy, you should probably be using the
    scipy.integrate tools rather than this function.
    """
    yout = y0.new_zeros((len(t), *y0.shape))

    yout[0] = y0


    for i in np.arange(len(t) - 1):

        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = derivs(y0, thist, *args, **kwargs)
        k2 = derivs(y0 + dt2 * k1, thist + dt2, *args, **kwargs)
        k3 = derivs(y0 + dt2 * k2, thist + dt2, *args, **kwargs)
        k4 = derivs(y0 + dt * k3, thist + dt, *args, **kwargs)
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout
