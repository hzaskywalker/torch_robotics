import torch
import numpy as np
from ..arith import dot, transpose


def dense_contact_dynamics(engine, jac, invM, tau, dist, velocity, contact_dof=1):
    """
    :param engine: the original engine... ideally we don't need that
    :param jac: (J: (n_j, 6,6), constraint_id (n_j,) <= n_c, obj_id (n_j) <= total_objects...
    :param invM: (obj, 6, 6)
    :param tau (obj, 6):
    :param N_b: number of object per scene
    :return:
    """
    jac, jac_id_c, jac_id_o = jac
    jac_batch, jac_o = engine.rigid_body_index2xy(jac_id_o)
    batch_size = engine.batch_size

    # ideally we should try to find the maximum number kinematics chain
    # and then solve the constraits inside the chain
    # a carefully optimized LCP should directly do the projected Gauss Siedel in the sparse form
    # currently I am not going to work for that as it's a little complex

    # we first count the maximum number of contacts
    #   ideally we should use a for loop to find the connected group .. now we will only use the all objects and
    #   find the maximum number of contacts per scene

    n_b = engine.n_rigid_body
    vdof = invM.shape[-1]
    # O(num of contact loop)

    _jac_batch = jac_batch.detach().cpu().numpy()
    _jac_id_c = jac_id_c.detach().cpu().numpy()
    with torch.no_grad():
        # O(collision loop on the cpu)
        batch_contact_id = np.zeros((len(jac_batch),), dtype=np.int32) - 1  # contact id in batch
        batch_nc = np.zeros((batch_size,), dtype=np.int32)

        for j in range(len(_jac_batch)):
            batch_id = int(_jac_batch[j])
            contact_id = _jac_id_c[j]
            if batch_contact_id[contact_id] < 0:
                batch_contact_id[contact_id] = batch_nc[batch_id]
                batch_nc[batch_id] += 1
            else:
                break
    max_nc = int(batch_nc.max())

    _jac = jac[:, 3:4] if contact_dof == 1 else (jac[:, 3:] if contact_dof == 3 else jac[:, [3,0,1,2,4,5]])

    """
    The following part should equivalent to the latter:
    J = jac.new_zeros(batch_size, max_nc, n_b * vdof)
    d0 = jac.new_zeros(batch_size, max_nc)

    for j in range(len(jac)):
        contact_id = jac_id_c[j]
        batch_id = int(jac_batch[j])
        obj_id = int(jac_o[j])
        # ------------ we enforce it to be a very simple contact here
        # we only extract the normal direction
        contact_ = batch_contact_id[contact_id] * contact_dof
        J[batch_id, contact_:contact_+contact_dof, obj_id * vdof:(obj_id+1) * vdof] = _jac[j]
        d0[batch_id, batch_contact_id[contact_id]]= dist[contact_id]
    """

    batch_contact_id = torch.tensor(batch_contact_id, dtype=torch.long, device=jac.device)
    contact_id = (jac_batch * max_nc + batch_contact_id[jac_id_c])
    d0 = jac.new_zeros(batch_size * max_nc).scatter(dim=0, index=contact_id, src=dist[jac_id_c]).reshape(batch_size, max_nc)

    dimq = n_b * vdof
    index = contact_id[:, None] * contact_dof + torch.arange(contact_dof, device=jac.device)[None,:]

    obj_id = (torch.arange(vdof, device=jac.device)[None, :] + jac_o[:, None] * vdof)
    index = (index[..., None] * dimq + obj_id[:, None, :]).reshape(-1)
    
    J = jac.new_zeros(batch_size * max_nc * contact_dof * dimq)
    J = J.scatter(dim=0, index=index.to(jac.device), src=_jac.reshape(-1))

    J = J.reshape(batch_size, max_nc, contact_dof, dimq).transpose(1, 2).reshape(batch_size, max_nc * contact_dof, dimq)

    # this loop is very hard to avoid, we assume that the number of objects is limited ...
    # otherwise, we should use sparse matrix representation ...
    invM_ = invM.reshape(n_b, batch_size, vdof, vdof).transpose(0, 1)
    invM = jac.new_zeros(batch_size, dimq, dimq)
    for i in range(n_b):
        invM[:, i * vdof:(i + 1) * vdof, i * vdof:(i + 1) * vdof] = invM_[:, i]
    tau = tau.reshape(n_b, batch_size, vdof).transpose(0, 1).reshape(batch_size, dimq)
    velocity = velocity.reshape(n_b, batch_size, vdof).transpose(0, 1).reshape(batch_size, dimq)

    # A = JM^{-1}J^T
    # a0 = JM^{-1}a_0
    JinvM = dot(J, invM)
    A = dot(JinvM, transpose(J))
    v0 = dot(J, velocity)
    a0 = dot(JinvM, tau)
    return A, a0, v0, d0, J
