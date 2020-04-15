# the same to sapien validator
# However, I am going to find the mathematical model for the robot arm
import numpy as np
import transforms3d
from robot.envs.hyrule.rl_env import ArmReachWithXYZ
import torch
from robot import tr, U
from robot.model.arm.exp.sapien_validator import ArmModel

def togpu(x):
    return U.togpu(x, dtype=torch.float64)


def get_env_agent(timestep=0.00001):
    env = ArmReachWithXYZ(timestep=timestep)
    seed = 2
    np.random.seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed) # goal sampling
    return env, env.agent


def pose2SE3(pose):
    p = pose.p
    q = pose.q
    mat = transforms3d.quaternions.quat2mat(q)

    out = np.zeros((4, 4))
    out[3, 3] = 1
    out[:3, :3] = mat
    out[:3, 3] = p
    return out

def solveJoint(agent, M):
    def calc(a, b):
        out = togpu(np.eye(4))[None,:]
        for j in range(a, b):
            out = tr.dot(out, togpu(M[j])[None, :])
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

            ee_T = togpu(pose2SE3(agent.get_ee_links().pose))[None, :]
            # A @ X @ B = ee_T
            X = tr.dot(tr.dot(tr.inv_trans(A), ee_T), tr.inv_trans(B))
            X[:, :3, :3] = tr.projectSO3(X[:, :3, :3])
            se3 = tr.logSE3(X)/theta
            vec = torch.cat((tr.so3_to_vec(se3[:, :3,:3]), se3[:,:3,3]), dim=-1)
            ans = vec + ans
            count += 1
        outs.append(ans/count)
    return torch.stack(outs)


def build_diff_model(env, timestep=0.0025, max_velocity=20, damping=0., dtype=torch.float64):
    model = ArmModel(7, gravity=[0, 0, -9.8],
                     max_velocity=max_velocity, timestep=timestep, damping=damping, dtype=dtype).cuda()
    M = []
    A = []
    G = []

    # seven dof
    agent = env.agent

    agent.set_qpos(agent.get_qpos()*0)
    def get_name(a):
        if a is None: return None
        return a.name

    """
    right_base_link
    right_shoulder_link
    right_arm_half_1_link
    right_arm_half_2_link
    right_forearm_link
    right_wrist_spherical_1_link
    right_wrist_spherical_2_link
    right_wrist_3_link
    right_ee_link
    """

    S = [] # location of cmass, execpt the first one, and the end-effector
    for i in agent.get_links()[1:]:
        cmass_pose = i.get_pose() * i.cmass_local_pose
        S.append(pose2SE3(cmass_pose))
    S.append(pose2SE3(agent.get_ee_links().get_pose()))

    S_prev = togpu(np.eye(4))[None,:]
    for s in S:
        s = togpu(s[None,:])
        M.append(tr.dot(tr.inv_trans(S_prev), s))
        S_prev = s
    M = torch.cat(M)

    G = []
    for i in agent.get_links()[1:]:
        out = np.zeros((6, 6))
        out[0, 0] = i.inertia[0]
        out[1, 1] = i.inertia[2]
        out[2, 2] = i.inertia[1]
        out[3, 3] = i.get_mass()
        out[4, 4] = i.get_mass()
        out[5, 5] = i.get_mass()
        G.append(togpu(out))
    G = torch.stack(G)


    #w, q = [0, 0, 1], [0, 0.25, 0]
    #screw1 = w + (-np.cross(w, q)).tolist()
    #A = torch.tensor([screw1, screw1], dtype=dtype, device='cuda:0')
    A = []
    for i in agent.get_joints():
        # T_p x = T_c y => y=inv(T_c) @ T_p @ x
        l = i.get_child_link()
        q = i.get_pose_in_child_frame().q
        pose = l.cmass_local_pose.inv() * i.get_pose_in_child_frame()

        #w, q = [0, 0, 1], [0, 0.25, 0]
        w = transforms3d.quaternions.rotate_vector([1, 0, 0], pose.q)
        w /= np.linalg.norm(w)
        q = pose.p
        screw = np.concatenate((w, (-np.cross(w, q))))
        A.append(screw)
    A = togpu(A)
    #print(A)
    #A2 = solveJoint(agent, M)
    #print(A2)

    model.assign(A, M, G)
    return model


def test_env_model():
    env, agent = get_env_agent()
    model = build_diff_model(env)



if __name__ == '__main__':
    test_env_model()
