# the same to sapien validator
# However, I am going to find the mathematical model for the robot arm
import numpy as np
import transforms3d
from robot.envs.hyrule.rl_env import ArmReachWithXYZ
import torch
from robot import tr, U
import robot.model.arm.exp.sapien_validator as sapien_validator

def check(a, b, m="", eps=1e-6):
    diff = np.abs(a - b).max()
    assert diff < eps, f"{m} {a}, {b}          max diff: {diff}"

def relative_check(a, b, m="", eps=1e-6):
    diff = (np.abs(a - b)/np.abs(b).clip(1, np.inf)).max()
    assert diff < eps, f"{m} {a}, {b}          max relative diff: {diff}"


def get_env_agent(timestep=0.00005):
    env = ArmReachWithXYZ(timestep=timestep, frameskip=1)
    seed = 2
    np.random.seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed) # goal sampling
    return env, env.agent


def solveJoint(agent, M):
    def calc(a, b):
        out = tr.togpu(np.eye(4))[None,:]
        for j in range(a, b):
            out = tr.dot(out, tr.togpu(M[j])[None, :])
        return out

    outs = []
    for i in range(len(M)-1):
        A = calc(0, i+1)
        B = calc(i+1, len(M))
        tmp = []
        N = 10
        #for j in range(N):
        #    theta = np.pi * 2 / N * j - np.pi
        ans = 0
        count = 0
        for theta in [np.pi/2, np.pi, -np.pi/2]:
            theta = np.random.random() * np.pi/2 + np.pi/2
            qpos = agent.get_qpos() * 0
            qpos[i] = theta
            agent.set_qpos(qpos)

            ee_T = tr.togpu(tr.pose2SE3(agent.get_ee_links().pose))[None, :]
            # A @ X @ B = ee_T
            X = tr.dot(tr.dot(tr.inv_trans(A), ee_T), tr.inv_trans(B))
            X[:, :3, :3] = tr.projectSO3(X[:, :3, :3])
            se3 = tr.logSE3(X)/theta
            vec = torch.cat((tr.so3_to_vec(se3[:, :3,:3]), se3[:,:3,3]), dim=-1)
            ans = vec + ans
            count += 1
        outs.append(ans/count)
    return torch.stack(outs)


def build_diff_model(env, timestep=0.0025, max_velocity=200, damping=0.0, dtype=torch.float64):
    model = sapien_validator.ArmModel(7, gravity=[0, 0, -9.8],
                     max_velocity=max_velocity, timestep=timestep, damping=damping, dtype=dtype).cuda()
    M = []
    A = []
    G = []

    # seven dof
    agent = env.agent

    agent.set_qpos(agent.get_qpos()*0)

    S, G = [], [] # location of cmass, execpt the first one, and the end-effector
    for i in agent.get_links()[1:]:
        cmass_pose = i.get_pose() * i.cmass_local_pose
        S.append(tr.pose2SE3(cmass_pose))

        out = np.zeros((6, 6))
        out[0, 0] = i.inertia[0]
        out[1, 1] = i.inertia[1]
        out[2, 2] = i.inertia[2]
        out[3, 3] = i.get_mass()
        out[4, 4] = i.get_mass()
        out[5, 5] = i.get_mass()
        G.append(tr.togpu(out))
    G = torch.stack(G)

    S.append(tr.pose2SE3(agent.get_ee_links().get_pose()))

    S_prev = tr.togpu(np.eye(4))[None,:]
    for s in S:
        s = tr.togpu(s[None,:])
        M.append(tr.dot(tr.inv_trans(S_prev), s))
        S_prev = s
    M = torch.cat(M)


    #w, q = [0, 0, 1], [0, 0.25, 0]
    #screw1 = w + (-np.cross(w, q)).tolist()
    #A = torch.tensor([screw1, screw1], dtype=dtype, device='cuda:0')
    A = []
    for i, l in zip(agent.get_joints(), agent.get_links()[1:]):
        # T_p x = T_c y => y=inv(T_c) @ T_p @ x
        pose = l.cmass_local_pose.inv() * i.get_pose_in_child_frame()

        w = transforms3d.quaternions.rotate_vector([1, 0, 0], pose.q)
        w /= np.linalg.norm(w)
        q = pose.p
        screw = np.concatenate((w, (-np.cross(w, q))))
        A.append(screw)
    A = tr.togpu(A)
    model.assign(A, M, G)
    return model


def test_fk():
    env, agent = get_env_agent()
    model = build_diff_model(env)

    for i in range(10):
        q = np.random.random(size=(7,)) * np.pi * 2 - np.pi
        agent.set_qpos(q)
        ee_label = tr.pose2SE3(agent.get_ee_links().pose)
        predict_ee = U.tocpu(model.fk(tr.togpu(q)[None, :], dim=0)[0])
        check(ee_label, predict_ee, "test_fk")
    print("passed fk test")

def test_inverse_dynamics():
    env, agent = get_env_agent()

    model = build_diff_model(env)

    #q, dq = [-2.148633, 2.129848], [-2.0222425, 6.676592 ]
    np.random.seed(1)
    q = np.random.random(size=(7,)) * np.pi/3 * 2 - np.pi/3
    dq = np.random.random(size=(7,)) * 10 - 5

    torque = np.random.random(size=(7,)) * 10 - 5

    q_torch = tr.togpu(q)[None,:]
    dq_torch = tr.togpu(dq)[None,:]
    torque_torch = tr.togpu(torque)[None,:]

    M = sapien_validator.compute_mass_matrix(agent, q)
    M2 = U.tocpu(model.compute_mass_matrix(q_torch)[0])
    check(M, M2, "compute mass")

    N_ = sapien_validator.compute_gravity(agent, q, dq)
    qacc = sapien_validator.compute_qacc(env, agent, q, dq, torque)
    N = sapien_validator.compute_gravity(agent, q, dq)
    img = env.render(mode='rgb_array')
    import cv2
    cv2.imwrite('x.jpg', img)
    check(N, N_, 'gravity')

    N2 = U.tocpu(model.compute_gravity(q_torch)[0])
    check(N, N2, "compute graivty", eps=1e-5)

    C = sapien_validator.compute_coriolis_centripetal(agent, q, dq)
    C2 = U.tocpu(model.compute_coriolis_centripetal(q_torch, dq_torch)[0])
    check(C, C2, "compute coriolis", eps=1e-5)

    qacc_torch = tr.togpu(qacc)[None, :]

    I = sapien_validator.inverse_dynamics(agent, q, dq, qacc, with_passive=True, external=True)
    I2 = U.tocpu(model.inverse_dynamics(q_torch, dq_torch, qacc_torch))
    check(I, I2, "inverse dynamics", eps=1e-4)
    print('inversed', I)
    # ----------------------------- QACC ----------------------------------


    qacc = sapien_validator.compute_qacc(env, agent, q, dq, torque)
    predict_qacc = U.tocpu(model.qacc(q_torch, dq_torch, torque_torch/50, damping=False)[0])

    check(qacc, predict_qacc, "qacc", eps=1e-5)


if __name__ == '__main__':
    #test_fk()
    test_inverse_dynamics()
