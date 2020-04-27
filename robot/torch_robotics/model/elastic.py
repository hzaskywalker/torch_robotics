# the code to form elastic collision for impulse based stepping system
# this is equivalent to that  n^T \cdot (v^{+}+v^{-})=0

import torch
import numpy as np
# from torch_geometric.utils import scatter_
from ..arith import dot, transpose
from ..solver.lcp import ProjectedGaussSiedelLCPSolver

class ElasticImpulse:
    def __init__(self, alpha0):
        self.alpha0 = alpha0
        self.solver = ProjectedGaussSiedelLCPSolver()

    def __call__(self, engine, jac, invM, tau, dist, velocity):
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
            contact_id = np.array((len(c_batch),), dtype=np.int32) # contact id in batch
            batch_nc = np.array((batch_size,), dtype=np.int32)
            for j in range(len(c_batch)):
                batch_id = int(c_batch[j])
                contact_id[j] = batch_nc[batch_id]
                batch_nc[batch_id] += 1

        max_nc = batch_nc.max()
        J = jac.new_zeros(batch_size, max_nc, n_b * vdof)
        d0 = jac.new_zeros(batch_size, max_nc)
        for j in range(len(jac)):
            contact = jac_id_c[j]
            batch_id = int(c_batch[contact])
            obj_id = int(c_o[contact])

            #------------ we enforce it to be a very simple model here
            J[batch_id, contact_id[contact], obj_id*vdof:(obj_id+1)*vdof] = jac[j, 3] # we only extract the normal direction
            d0[batch_id, contact_id[contact]] = dist[contact]

        invM_ = invM.reshape(n_b, batch_size, vdof, vdof).transpose(0, 1)
        invM = jac.new_zeros(batch_size, n_b * vdof, n_b * vdof)
        for i in range(n_b):
            invM[:, i*vdof:(i+1)*vdof, i*vdof:(i+1)*vdof] = invM_[:, i]

        tau = tau.reshape(n_b, batch_size, vdof).transpose(0, 1)

        velocity = velocity.reshape(n_b, batch_size, vdof).transpose(0, 1).reshpae(batch_size, n_b, -1)

        # A = JM^{-1}J^T
        # a0 = JM^{-1}a_0
        JinvM = dot(J, invM)
        A = dot(JinvM, transpose(J))
        v0 = dot(J, velocity)
        a0 = dot(JinvM, tau)

        # we now have the following matrix
        #   Af+a0=a
        #   (v1-v0) = A hf + ha0
        #   d1 = d0 + h v1


        # Force: f should be greater than zero
        # LCP: f\dot (d-\alpha_0) = 0, e.g.,
        #   if f\neq 0, they must contact with each other
        #   if d>\alpha_0, then there should be no force
        # Separation: d1 \ge \alpha_0
        # Elastic collision: v1\ge -v_0; after the collision, the velocity should reverse the sign
        # v1 = (d1-d0)/h >= -v0 => d1 \ge -hv0 + d0
        # we also have d1 >= alpha0
        # d1 = h*h(Af+a0) + hv0 + d0

        h = engine.dt
        alpha0 = self.alpha0
        X = h*h * A
        Y = h*h * a0 + h*v0 + d0 - (d0-h*v0).clamp(alpha0)

        f = self.solver(X, Y) #(batch, nc)

        # after we have solved the f, what shall we do for the next?
        # calculate J^Tf
        a1 = dot(transpose(J), f).reshape(batch_size * n_b, vdof)
        return a1
