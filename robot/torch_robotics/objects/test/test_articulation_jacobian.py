import torch
import numpy as np
from robot.torch_robotics import arith as tr, engine2

def finite_difference(link_id, pose, qpos, M, A, epsilon=1e-4, type='body'):
    assert M.shape[0] == 1

    oldT = tr.fk_in_space(qpos, M, A)
    pose_in_body = tr.dot(tr.inv_trans(oldT[:, link_id]), pose)

    qpos = qpos[0]
    n = M.shape[1] - 1

    todo = []
    for i in range(n):
        x = qpos.clone()
        x[i] = x[i] + epsilon
        todo.append(x)
        x = qpos.clone()
        x[i] = x[i] - epsilon
        todo.append(x)
    todo = torch.stack(todo)
    M = M.expand(todo.shape[0], -1, -1, -1)
    A = A.expand(todo.shape[0], -1, -1)

    T = tr.fk_in_space(todo, M, A)

    new_pose = tr.dot(T[:, link_id], pose_in_body.expand(T.shape[0], -1, -1))

    # this affect if we calculate the body jacobian or spatial jacobian..
    if type == 'body':
        velocity = tr.logSE3(tr.dot(tr.inv_trans(pose).expand(new_pose.shape[0], -1, -1), new_pose))
    else:
        velocity = tr.logSE3(tr.dot(new_pose, tr.inv_trans(pose).expand(new_pose.shape[0], -1, -1)))
    velocity = tr.se3_to_vec(velocity).reshape(n, 2, 6)
    velocity = (velocity[:, 0] - velocity[:, 1])/2/epsilon
    return velocity.transpose(0, 1)


def test():
    sim = engine2.Engine2()

    robot = sim.robot(None)[0]
    M, A = robot.M, robot.A

    link_id = 2
    robot.qpos[:] = tr.togpu([np.pi/4, np.pi/3])
    pose = robot.fk()[:, -1]
    pose = tr.togpu([
       [
           [1, 0, 0, 1],
           [0, 1, 0, 2],
           [0, 0, 1, -2],
           [0, 0, 0, 1],
       ]
    ])
    qpos = robot.qpos
    jac_body = finite_difference(link_id, pose, qpos, M, A, type='body')

    link_id = tr.togpu([link_id]).long()
    predict_jac = robot.compute_jacobian(link_id, pose)
    assert (predict_jac[0] - jac_body).abs().max() < 1e-6

if __name__ == '__main__':
    test()
