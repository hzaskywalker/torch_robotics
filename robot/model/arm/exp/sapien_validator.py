# the goal is to understand the model behind sapien
# the basic idea is for very short timestep, check if the compute_inverse_dynamics is correct
# and then use it to recover the motion equation
# and then recover the integral
import numpy as np
import torch
from robot import A, U, tr

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

def sample():
    env = A.envs.acrobat.GoalAcrobat(timestep=0.025, damping=0)
    for i in range(10):
        a = env.action_space.sample()
        obs = env.step(a)[0]
    o = obs['observation']
    print(o[:2], o[2:4], a, env.agent.get_qacc())


def get_env_agent():
    env = A.envs.acrobat.GoalAcrobat(timestep=0.00001, damping=0, clip_action=False)
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
    torque = [5.5375643, 28.926117]
    ddq = np.array([25.347101, 207.08884])

    M = compute_mass_matrix(agent, q)
    from robot.model.arm.exp.qacc import build_diff_model
    from robot.model.arm.exp.phys_model import ArmModel

    model: ArmModel = build_diff_model(env)
    q_torch = U.togpu(q, dtype=torch.float64)[None,:]
    dq_torch = U.togpu(dq, dtype=torch.float64)[None,:]
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


if __name__ == '__main__':
    #sample()
    #test_inverse_dynamics()
    test_compute_all()
