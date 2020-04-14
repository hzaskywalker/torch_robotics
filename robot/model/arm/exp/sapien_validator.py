# the goal is to understand the model behind sapien
# the basic idea is for very short timestep, check if the compute_inverse_dynamics is correct
# and then use it to recover the motion equation
# and then recover the integral
import numpy as np
import torch
from robot import A, U, tr
from robot.model.arm.exp.phys_model import ArmModel

def build_diff_model(env, timestep=0.01, max_velocity=100, damping=0., dtype=torch.float64):
    model = ArmModel(2, max_velocity=max_velocity, timestep=timestep, damping=damping, dtype=dtype).cuda()

    M = torch.tensor(
        np.array(
            [[[1, 0, 0, 0],[0, 1, 0, -0.25], [0, 0, 1, 0], [0, 0, 0, 1]],
             [[1, 0, 0, 0], [0, 1, 0, -0.5], [0, 0, 1, 0], [0, 0, 0, 1]],
             [[1, 0, 0, 0], [0, 1, 0, -0.25], [0, 0, 1, 0], [0, 0, 0, 1]],]
        ), dtype=dtype, device='cuda:0')

    w, q = [0, 0, 1], [0, 0.25, 0]
    screw1 = w + (-np.cross(w, q)).tolist()
    A = torch.tensor([screw1, screw1], dtype=dtype, device='cuda:0')
    G = []
    for idx, i in enumerate(env.unwrapped.model.get_links()[1:]):
        out = np.zeros((6, 6))
        out[0, 0] = i.inertia[0]
        out[1, 1] = i.inertia[2]
        out[2, 2] = i.inertia[1]
        out[3, 3] = i.get_mass()
        out[4, 4] = i.get_mass()
        out[5, 5] = i.get_mass()
        G.append(torch.tensor(out, dtype=dtype, device='cuda:0'))
    G = torch.stack(G)
    model.assign(A, M, G)
    return model

def inverse_dynamics(agent, q, dq, ddq, with_passive=True, external=False):
    # NEED to check if we need step for the inverse_dynamcis
    # external will add friction
    agent.set_qpos(q)
    agent.set_qvel(dq)
    out = agent.compute_inverse_dynamics(ddq)
    if with_passive:
        out += agent.compute_passive_force(external=external)
    return out

def compute_mass_matrix(agent, q):
    n = len(q)
    M = np.zeros((n, n))
    for i in range(len(q)):
        dq = np.zeros_like(q)
        ddq = np.zeros_like(q)
        ddq[i] = 1
        M[:, i] = inverse_dynamics(agent, q, dq, ddq, with_passive=False) # the return is Mddq
    return M

def compute_coriolis_centripetal(agent, q, dq):
    agent.set_qpos(q)
    agent.set_qvel(dq)
    return agent.compute_passive_force(external=False, gravity=False, coriolisAndCentrifugal=True)

def compute_gravity(agent, q, dq):
    agent.set_qpos(q)
    agent.set_qvel(dq)
    return agent.compute_passive_force(external=False, gravity=True, coriolisAndCentrifugal=False)

def compute_external(agent, q, dq):
    agent.set_qpos(q)
    agent.set_qvel(dq)
    return agent.compute_passive_force(external=True, gravity=False, coriolisAndCentrifugal=False)

def compute_qacc(env, agent, q, dq, torque):
    assert env.dt < 1e-4
    agent.set_qpos(q)
    agent.set_qvel(dq)
    env.step(torque/50)
    return agent.get_qacc()

def compute_forward(env, agent, q, dq, qf, timestep):
    k = timestep/env.timestep
    agent.set_qpos(q)
    agent.set_qvel(dq)
    print(np.round(k))
    for i in range(int(np.round(k))):
        env.step(qf/50)
    return agent.get_qpos(), agent.get_qvel(), agent.get_qacc()

def sample():
    env = A.envs.acrobat.GoalAcrobat(timestep=0.025, damping=0)
    for i in range(10):
        a = env.action_space.sample()
        obs = env.step(a)[0]
    o = obs['observation']
    print(o[:2], o[2:4], a, env.agent.get_qacc())


def get_env_agent(timestep=0.00001, damping=0):
    env = A.envs.acrobat.GoalAcrobat(timestep=timestep, damping=damping, clip_action=False)
    seed = 2
    np.random.seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)
    return env, env.agent


def test_inverse_dynamics():
    env, agent = get_env_agent()

    q, dq = [-2.148633, 2.129848], [-2.0222425, 6.676592 ]
    a = np.array([0.31899363, 0.5958361])
    ddq = np.array([25.347101, 207.08884])

    torque = a * 50
    print('fake torque', torque)

    print('q:', q, 'dq:', dq, 'ddq:', ddq)
    inversed_torque = inverse_dynamics(agent, q, dq, ddq)
    agent.set_qpos(q)
    agent.set_qvel(dq)
    env.step(inversed_torque/50)
    print(inversed_torque, agent.get_qacc())

def test_compute_all():
    env, agent = get_env_agent()

    q, dq = np.array([-2.148633, 2.129848]), np.array([-2.0222425, 6.676592])
    torque = np.array([5.5375643, 28.926117])
    ddq = np.array([25.347101, 207.08884])

    M = compute_mass_matrix(agent, q)

    model: ArmModel = build_diff_model(env)
    q_torch = U.togpu(q, dtype=torch.float64)[None,:]
    dq_torch = U.togpu(dq, dtype=torch.float64)[None,:]
    ddq_torch = U.togpu(ddq, dtype=torch.float64)[None,:]
    torque_torch = U.togpu(torque, dtype=torch.float64)[None,:]

    model_M = U.tocpu(model.compute_mass_matrix(q_torch)[0])
    print("DIFF BETWEEN TWO MASS", M - model_M)

    # check gravity
    print(q.shape, dq.shape)
    gravity = compute_gravity(agent, q, dq)
    print("GRAVITY", gravity, U.tocpu(model.compute_gravity(q_torch)[0]))

    C = compute_coriolis_centripetal(agent, q, dq)
    model_C = U.tocpu(model.compute_coriolis_centripetal(q_torch, dq_torch))[0]
    print("C", C, model_C)

    print('external', compute_external(agent, q, dq))

    print('sapien inverse', inverse_dynamics(agent, q, dq, ddq))
    print('torch inverse', model.inverse_dynamics(q_torch, dq_torch, ddq_torch)) # /50

    print('sapien qacc', compute_qacc(env, agent, q, dq, torque))
    print('torch qacc', model.qacc(q_torch, dq_torch, torque_torch/50))


def test_integral():
    #damping=0.5
    damping = 0.5

    #total = 0.5
    total = 0.1
    model_dt = 0.025
    sapien_dt = 0.025

    env, agent = get_env_agent(timestep=sapien_dt, damping=damping)

    model: ArmModel = build_diff_model(env, timestep=model_dt, max_velocity=np.inf, damping=damping)

    q, dq = np.array([-2.148633, 1.129848]), np.array([-10.0222425, 6.676592])
    print('initial q, dq', q, dq)
    #torque = np.array([10.5375643, 28.926117]) * 5
    torque = np.array([20.5375643, 20.926117])

    q1, dq1, qacc1 = compute_forward(env, agent, q, dq, torque, total)
    print((q1+np.pi)%(np.pi*2)-np.pi, dq1, qacc1)

    state = U.togpu(np.concatenate((q, dq), axis=-1), dtype=torch.float64)[None,:]
    torque_torch = U.togpu(torque, dtype=torch.float64)[None,:]

    for i in range(int(np.round(total/model_dt))):
        #f = torque_torch - damping * state[:,2:4]
        f = torque_torch
        state = model.forward(state, f/50)[0]

    state1 = U.tocpu(state[0])
    print(state1, model.qacc(state[...,:2], state[...,2:], torque_torch/50)[0])



if __name__ == '__main__':
    #sample()
    test_inverse_dynamics()
    test_compute_all()
    #test_integral()
