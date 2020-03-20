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
    Example Input:
        z = -1e-7
    Output:
        True
    """
    return z.abs() < 1e-6

def make_matrix(arrays):
    # all elements of arrays has the same shape
    return torch.cat([torch.cat(i, dim=-1) for i in arrays], dim=-2)

def vec_to_so3(omg):
    """Converts a 3-vector to an so(3) representation
    :param omg: A 3-vector
    :return: The skew symmetric representation of omg
    Example Input:
        omg = np.array([1, 2, 3])
    Output:
        np.array([[ 0, -3,  2],
                  [ 3,  0, -1],
                  [-2,  1,  0]])
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
    omgmat = safe_div(so3mat, theta)

    theta = theta[..., None, None]
    Rodrigue =  eye + torch.sin(theta) * omgmat \
            + (1-torch.cos(theta)) * dot(omgmat, omgmat)

    mask = mask[..., None, None].float()
    return mask * eye + (1-mask) * Rodrigue # if near zero, just return identity...

def Rp_to_trans(R, p):
    return make_matrix((
        (R, p[..., None]),
        (torch.zeros_like(R[..., -1:, :]), torch.ones_like(p[..., -1:, None]))
    ))


def expse3(se3mat):
    """Computes the matrix exponential of an se3 representation of
    exponential coordinates
    :param se3mat: A matrix in se3
    :return: The matrix exponential of se3mat

    page 105 on the book
    """
    omgtheta = so3_to_vec(se3mat[..., 0: 3, 0: 3])
    is_translation = NearZero(torch.norm(omgtheta, dim=-1))

    eye = eyes_like(se3mat, 3)
    theta = torch.norm(omgtheta, dim=-1) #(b,)

    omgmat = safe_div(se3mat[...,:3,:3], theta)
    v = safe_div(se3mat[..., :3, 3], theta)

    R2 = expso3(se3mat[..., :3, :3])

    theta = theta[..., None, None]
    tmp = eye * theta + (1-torch.cos(theta)) * omgmat + (theta - torch.sin(theta)) * dot(omgmat, omgmat)
    p2 = dot(tmp, v)
    out0 = Rp_to_trans(R2, p2)
    out1 = Rp_to_trans(eye, se3mat[..., 0:3, 3])

    is_translation = is_translation[..., None, None].float()
    return is_translation * out1 + (1-is_translation) * out0


def inv_trans(T):
    """Inverts a homogeneous transformation matrix
    :param T: A homogeneous transformation matrix
    :return: The inverse of T
    Uses the structure of transformation matrices to avoid taking a matrix
    inverse, for efficiency.
    Example input:
    Output:
        np.array([[1,  0, 0,  0],
                  [0,  0, 1, -3],
                  [0, -1, 0,  0],
                  [0,  0, 0,  1]])
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
    return make_matrix((
        (R, zero),
        (dot(vec_to_so3(p), R), R)
    ))

def ad(V):
    """Calculate the 6x6 matrix [adV] of the given 6-vector
    :param V: A 6-vector spatial velocity
    :return: The corresponding 6x6 matrix [adV]
    Used to calculate the Lie bracket [V1, V2] = [adV1]V2
    Example Input:
        V = np.array([1, 2, 3, 4, 5, 6])
    Output:
        np.array([[ 0, -3,  2,  0,  0,  0],
                  [ 3,  0, -1,  0,  0,  0],
                  [-2,  1,  0,  0,  0,  0],
                  [ 0, -6,  5,  0, -3,  2],
                  [ 6,  0, -4,  3,  0, -1],
                  [-5,  4,  0, -2,  1,  0]])
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


def inverse_dynamics(theta, dtheta, ddtheta, gravity, Ftip, M, G, S):
    """Computes inverse dynamics in the space frame for an open chain robot
    :param thetalist: n-vector of joint variables
    :param dthetalist: n-vector of joint rates
    :param ddthetalist: n-vector of joint accelerations
    :param g: Gravity vector g
    :param Ftip: Spatial force applied by the end-effector expressed in frame
                 {n+1}
    :param Mlist: List of link frames {i} relative to {i-1} at the home
                  position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame, in the format
                  of a matrix with axes as the columns
    :return: The n-vector of required joint forces/torques
    This function uses forward-backward Newton-Euler iterations to solve the
    equation:
    taulist = Mlist(thetalist)ddthetalist + c(thetalist,dthetalist) \
              + g(thetalist) + Jtr(thetalist)Ftip
    """
    batch_shape = theta.shape[:-1]
    n = theta.shape[-1]
    #Mi = torch.eye(4)
    Mi = eyes_like(M[..., 0, :, :], 4) #

    Vi = theta.new_zeros(batch_shape + (6,)) # initial Vi

    Vdi = theta.new_zeros(batch_shape + (6,)) # initial Vi
    Vdi[..., -3:] = gravity # g is the gravity acceleration

    AdT = []
    A = []
    V = []
    dV = []
    for i in range(n):
        Mi = dot(Mi, M[:, i])
        Ai = dot(Adjoint(inv_trans(Mi)), S[:, i])
        # T_{i, i-1} = e^{-[A_i]\theta_i}M_{i-1,i}^-1
        Ti = dot(expse3(vec_to_se3(Ai * -theta[:, i, None])), inv_trans(M[:, i]))
        AdTi = Adjoint(Ti)
        Vi = Ai * dtheta[:, i] +dot(AdTi, Vi) # new Vi
        dVi= dot(AdTi, Vdi) + Ai * ddtheta[:, i] + dot(ad(Vi), Ai) * ddtheta[:, i]

        AdT.append(AdTi); dV.append(dVi); V.append(Vi); A.append(Ai)

    AdT.append(Adjoint(inv_trans(M[:, n]))) # the
    Fi = Ftip
    tau = []
    for i in range(n-1, -1, -1):
        #F = dot(transpose(AdT[i+1]), Fi)
        Fi = newton_law(G[:, i], V[i], dV[i]) + dot(transpose(AdT[i + 1]), Fi)
        # Fi^TA
        tau.append((Fi * A[i]).sum(dim=-1))

    return torch.stack(tau[::-1], dim=-1)
