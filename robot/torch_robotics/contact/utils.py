import torch
import numpy as np
from ..arith import dot, transpose


def dense_contact_dynamics(engine, jac, invM, tau, dist, velocity):
    """
    :param engine: the original engine... ideally we don't need that
    :param jac: (J: (n_j, 6,6), constraint_id (n_j,) <= n_c, obj_id (n_j) <= total_objects...
    :param invM: (obj, 6, 6)
    :param tau (obj, 6):
    :param N_b: number of object per scene
    :return:
    """
    jac, jac_id_c, jac_id_o = jac
    c_batch, c_o = engine.rigid_body_index2xy(jac_id_o)
    batch_size = engine.batch_size

    # ideally we should try to find the maximum number kinematics chain
    # and then solve the constraits inside the chain
    # a carefully optimized LCP should directly do the projected Gauss Siedel in the sparse form
    # currently I am not going to work for that as it's

    # step 1: count the maximum number of contacts, and the maximum of the objects in one connected group
    #   ideally we should use a for loop to find the connected group .. now we will only use the all objects and
    #   find the maximum number of contacts per scene

    n_b = engine.n_rigid_body
    vdof = invM.shape[-1]

    # O(num of contact loop)
    with torch.no_grad():
        batch_contact_id = np.zeros((len(c_batch),), dtype=np.int32)  # contact id in batch
        batch_nc = np.zeros((batch_size,), dtype=np.int32)
        for j in range(len(c_batch)):
            batch_id = int(c_batch[j])
            batch_contact_id[j] = batch_nc[batch_id]
            batch_nc[batch_id] += 1

    max_nc = batch_nc.max()
    J = jac.new_zeros(batch_size, max_nc, n_b * vdof)
    d0 = jac.new_zeros(batch_size, max_nc)
    for j in range(len(jac)):
        contact_id = jac_id_c[j]
        batch_id = int(c_batch[contact_id])
        obj_id = int(c_o[contact_id])

        # ------------ we enforce it to be a very simple contact here
        J[batch_id, batch_contact_id[contact_id], obj_id * vdof:(obj_id + 1) * vdof] = jac[
            j, 3]  # we only extract the normal direction
        d0[batch_id, batch_contact_id[contact_id]] = dist[contact_id]

    invM_ = invM.reshape(n_b, batch_size, vdof, vdof).transpose(0, 1)
    invM = jac.new_zeros(batch_size, n_b * vdof, n_b * vdof)
    for i in range(n_b):
        invM[:, i * vdof:(i + 1) * vdof, i * vdof:(i + 1) * vdof] = invM_[:, i]

    tau = tau.reshape(n_b, batch_size, vdof).transpose(0, 1).reshape(batch_size, n_b * vdof)
    velocity = velocity.reshape(n_b, batch_size, vdof).transpose(0, 1).reshape(batch_size, n_b * vdof)

    # A = JM^{-1}J^T
    # a0 = JM^{-1}a_0
    JinvM = dot(J, invM)
    A = dot(JinvM, transpose(J))
    v0 = dot(J, velocity)
    a0 = dot(JinvM, tau)

    return A, a0, v0, d0, J
