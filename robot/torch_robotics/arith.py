# torch robotics library
# simulate a robot arm with pytorch
# this is mostly based on the modern_robotics.py

import torch
import numpy as np

#-----------------------  Kinematics -------------------

def trans_to_Rp(T):
    """Converts a homogeneous transformation matrix into a rotation matrix
    and position vector
    :param T: A homogeneous transformation matrix
    :return R: The corresponding rotation matrix,
    :return p: The corresponding position vector.
    """
    return T[..., 0: 3, 0: 3], T[..., 0: 3, 3]


def transpose(R):
    return R.transpose(-1, -2)


def translate(p):
    eye = eyes_like(p[..., None,:])
    return Rp_to_trans(eye, p)


def dot(A, B):
    assert A.shape[0] == B.shape[0]
    if A.dim() == B.dim():
        return A @ B
    elif A.dim() == B.dim() + 1:
        return (A @ B[..., None])[..., 0]
    else:
        raise NotImplementedError(f"can'd dot product: A.shape: {A.shape}, B.shape: {B.shape}")


def normalize(V):
    """Normalizes a vector
    :param V: A vector
    :return: A unit vector pointing in the same direction as z
    """
    return V / torch.norm(V, dim=-1, keepdim=True)


def NearZero(z):
    """Determines whether a scalar is small enough to be treated as zero
    :param z: A scalar input to check
    :return: True if z is close to zero, false otherwise
    """
    return z.abs() < 1e-10


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
    return so3mat[..., [2, 0, 1], [1,2,0]]


def vec_to_se3(V):
    """Converts a spatial velocity vector into a 4x4 matrix in se3
    :param V: A 6-vector representing a spatial velocity
    :return: The 4x4 se3 representation of V
    """
    #return np.r_[np.c_[VecToso3([V[0], V[1], V[2]]), [V[3], V[4], V[5]]],
    #             np.zeros((1, 4))]
    return make_matrix((
        [vec_to_so3(V[..., :3]), V[..., 3:, None]],
        [V.new_zeros((*V.shape[:-1], 1, 4)), ]
    ))


def safe_div(a, b):
    b = b + (torch.abs(b) < 1e-15).float() * 1e-14 # make sure it's not zero
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


def trace(R):
    return (R * eyes_like(R)).sum(dim=(-1, -2))

def projectSO3(R):
    # project a matrix into SO3, by 6d pose methods..
    out = torch.zeros_like(R)
    a1, a2 = R[..., 0], R[..., 1]
    assert a1.norm(dim=-1).min() > 1e-15, "check init norm"

    b1 = normalize(a1)
    b2 = normalize(a2 - (a2 * b1).sum(dim=-1, keepdim=True) * b1)
    b3 = torch.cross(b1, b2, dim=-1)
    out[..., 0], out[..., 1], out[..., 2] = b1, b2, b3
    return out


def logSO3(R):
    # NOTICE that log will produce NAN if the input is not in SO3
    # In this case please project it into SO3 for a reasonable result
    acosinput = (trace(R) - 1)/2
    condition1 = (acosinput >= 1.).float() # in this case it's zero

    #omg1 = (1.0 / np.sqrt(2 * (1 + R[2][2]))) * np.array([R[0][2], R[1][2], 1 + R[2][2]])
    condition2 = (acosinput <=-1).float() * (1-condition1)
    mat = R + eyes_like(R) # (0,0), (0,1), (0,2)
    to_div = 2 * torch.stack([R[..., i, i] for i in range(3)], dim=-1) + 2
    near_zero = NearZero(to_div).float()
    to_div += near_zero # we should never add a term to be zero..

    #TODO:very ugly need optimization..
    mask = (1-near_zero)
    mask[..., 0] *= (1-mask[..., 1]) * (1-mask[..., 2])
    mask[..., 1] *= (1-mask[..., 2])

    ans = ((mat / to_div[...,None,:].sqrt()) * mask[..., None,:]).sum(dim=-1)
    ans = vec_to_so3(np.pi * ans) * condition2[..., None, None]

    condition3 = (1-condition1) * (1-condition2)
    theta = torch.acos(acosinput.clamp(-1, 1))[...,None,None]
    sin = torch.sin(theta)
    sin += NearZero(sin).float() * 1e-15
    out = theta/2.0/sin * (R - transpose(R))
    ans += condition3[..., None, None] * out
    return ans


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
    theta = torch.stack([a, b, c]).norm(dim=0)

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


def logSE3(T):
    R, p = trans_to_Rp(T)
    omgmat = logSO3(R)

    equal_zero = (omgmat.abs().sum(dim=(-2, -1)) < 1e-12).float() # in this case, return zero, p

    theta = torch.acos(((trace(R) - 1) / 2.0).clamp(-1, 1))
    eye = eyes_like(R)
    theta_to_div = theta + (theta.abs() < 1e-15).float() * 1e-14
    tanh_theta = torch.tan(theta/2)
    tanh_theta = tanh_theta + (tanh_theta.abs() < 1e-15).float() * 1e-14
    p2 = dot(eye - omgmat/2. + (1.0 / theta_to_div - 1.0 / tanh_theta / 2)[..., None, None]\
             * dot(omgmat,omgmat)/theta_to_div[..., None, None], p)

    p2 = (1-equal_zero)[..., None] * p2 + p * equal_zero[..., None]
    omgmat = omgmat * (1-equal_zero)[..., None, None]
    out = p2.new_zeros((*omgmat.shape[:-2], 4, 4))
    out[...,:3,:3] = omgmat
    out[...,:3,3] = p2
    return out


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


def transform_vector(T, p):
    # named in a similar way to the rotate_vector in transform 3d
    _R, _p = trans_to_Rp(T)
    return dot(_R, p) + _p


def Adjoint(T):
    """Computes the adjoint representation of a homogeneous transformation
    matrix
    :param T: A homogeneous transformation matrix
    :return: The 6x6 adjoint representation [AdT] of T
    """
    R, p = trans_to_Rp(T)
    zero = R.new_zeros((*R.shape[:-2], 6, 6))
    zero[...,:3,:3] = R
    zero[...,3:,:3] = dot(vec_to_so3(p), R)
    zero[...,3:,3:] = R
    return zero


#-----------------------  Robot arm ---------------------

def fk_in_space(theta, M, A):
    """
    :param theta: qpos
    :param M: home transformation
    :param S: screw axis in space frame and the home configuration
    :return:
    """
    n = theta.shape[1]
    assert n == M.shape[1] - 1
    T = eyes_like(M[..., 0, :, :], 4)
    outputs = []
    for i in range(n):
        Ai = A[:, i]
        Ti = dot(M[:, i], expse3(vec_to_se3(Ai * theta[:, i, None])))
        T = dot(T, Ti)
        outputs.append(T)
    outputs.append(dot(T, M[:, n]))
    return torch.stack(outputs, dim=1)


def jacobian_space(theta, M, A):
    S = A_to_S(A, M)
    Js = transpose(S.clone())
    T = eyes_like(M[:, 0, :, :], 4)
    for i in range(1, theta.shape[-1]):
        T = dot(T, expse3(vec_to_se3(S[:, i-1]) * theta[:, i-1][:, None, None]))
        Js[:, :, i] = dot(Adjoint(T), S[:, i])
    return Js


def jacobian_body(theta, M, A):
    Js = jacobian_space(theta, M, A)
    T_sb = fk_in_space(theta, M, A)[:, -1]
    return dot(Adjoint(inv_trans(T_sb)), Js)


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

def adT(V):
    """Calculate the 6x6 matrix [adV] of the given 6-vector
    :param V: A 6-vector spatial velocity
    :return: The corresponding 6x6 matrix [adV]
    Used to calculate the Lie bracket [V1, V2] = [adV1]V2
    """
    #omgmat = VecToso3([V[0], V[1], V[2]])
    assert V.shape[-1] == 6
    omgmatT = -vec_to_so3(V[..., :3])
    vmatT = -vec_to_so3(V[..., 3:])
    adv = omgmatT.new_zeros(V.shape[:-1]+(6, 6))
    adv[..., :3,:3] = omgmatT
    adv[..., 3:,3:] = omgmatT
    adv[..., :3,3:] = vmatT
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


def A_to_S(A, M):
    Mi = eyes_like(M[:, 0, :, :], 4) #
    S = []
    for i in range(A.shape[1]):
        Mi = dot(Mi, M[:, i])
        Si = dot(Adjoint(Mi), A[:, i])
        S.append(Si)
    return torch.stack(S, dim=1)


def inverse_dynamics(theta, dtheta, ddtheta, gravity, Ftip, M, G, A):
    """Computes inverse dynamics in the space frame for an open chain robot
    :param Mlist: List of link frames {i} relative to {i-1} at the home
                  position, matrix of (b, n+1, 4, 4)
    :param Glist: Spatial inertia matrices Gi of the links, matrix of (b, n, 6, 6)
    :param Alist: Screw axes Si of the joints in frame {i}, matrix of (b, n, 6)
    :return: The n-vector of required joint forces/torques
    """
    batch_shape = theta.shape[:-1]
    assert len(batch_shape) == 1

    n = theta.shape[-1]
    assert n == A.shape[1], f"input dof doesn't match the model, the input shape:{n}, the parameter shape:{A.shape}"
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
        assert A[:, i].shape == Fi.shape
        tau.append((Fi * A[:, i]).sum(dim=-1))
        last_AdT = AdT[:, i]

    out = torch.stack(tau[::-1], dim=-1)
    return out


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


def compute_all_dynamic_parameters(theta, dtheta, gravity, Ftip, M, G, A):
    def expand_n(array, n):
        return array[:, None].expand(-1, n, *((-1,)*(array.dim()-1))).reshape(-1, *array.shape[1:])

    assert theta.dim() == 2

    b = theta.shape[0]
    n = theta.shape[-1]
    #ddtheta = eyes_like(M[..., 0, :, :], n=n).permute(1, 0, 2).reshape(-1, n) # n,
    #ddtheta = torch.eye(n, device=theta.device, dtype=theta.dtype)[None,:].expand(theta.shape[0], -1, -1)
    #ddtheta = ddtheta.permute(1, 0, 2).reshape(-1, n)
    ddtheta = theta.new_zeros((b, n+3, n))
    ddtheta[:,:n,:n] = torch.eye(n)[None,:].expand(b, -1, -1)
    ddtheta = ddtheta.reshape(-1, n)

    _dtheta = theta.new_zeros((b, n+3, n))
    _dtheta[:, -3] = dtheta
    _dtheta = _dtheta.reshape(-1, n)

    _gravity = theta.new_zeros((b, n+3, 3))
    _gravity[:,-2] = gravity
    _gravity = _gravity.reshape(-1, 3)

    _ftip = theta.new_zeros((b, n+3, 6))
    _ftip[:, -1] = Ftip
    _ftip = _ftip.reshape(-1, 6)

    theta = expand_n(theta, n+3)
    M = expand_n(M, n+3)
    G = expand_n(G, n+3)
    A = expand_n(A, n+3)

    all = inverse_dynamics(theta, _dtheta, ddtheta, _gravity, _ftip, M, G, A).reshape(-1, n+3, n)
    return all[:,:n], all[:, n], all[:, n+1], all[:, n+2]


def forward_dynamics(theta, dtheta, tau, gravity, Ftip, M, G, A):
    mass, c, g, f = compute_all_dynamic_parameters(theta, dtheta, gravity, Ftip, M, G, A)
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

def normal2pose(normal):
    normal = normalize(normal)
    one = torch.zeros_like(normal); one[..., 0] = 1
    two = torch.zeros_like(normal); two[..., 1] = 1
    mask = NearZero(((normal.abs() - one)**2).sum(dim=-1)).float()[:, None]
    r = one * (1-mask) + two * mask
    return projectSO3(torch.stack((normal, r, r), dim=-1))

def transform_wrench(F_a, T_ab):
    # transfrom wrench in coordinate T to
    # F_b = [Ad_{Tab}]^TF_a
    return dot(transpose(Adjoint(T_ab)), F_a)

# ------------------------- SAPIEN ---------------------------------

def togpu(x):
    from robot import U
    return U.togpu(x, dtype=torch.float64)

def totensor(x):
    from robot import U
    return torch.tensor(x, dtype=torch.float64)

def tocpu(x):
    from robot import U
    return U.tocpu(x)


def pose2SE3(pose):
    import transforms3d
    p = pose.p
    q = pose.q
    mat = transforms3d.quaternions.quat2mat(q)

    out = np.zeros((4, 4))
    out[3, 3] = 1
    out[:3, :3] = mat
    out[:3, 3] = p
    return out


# ------------------------- TODO: control -----------------------------

