import os
import argparse
import numpy as np
from robot.utils import togpu
import tqdm
os.environ['MUJOCO_GL'] = "osmesa" # for headless rendering
import gym
from robot.controller.mb_controller import MBController
from robot.model.gnn_forward import GNNForwardAgent
from robot.envs.dm_control import make as dm_make
from robot.utils import Visualizer
from robot.envs.dm_control.wrapper import GraphDmControlWrapper


def make(env_name, mode='Graph'):
    """
    :return: environment with graph, node attr and edge attr
    """
    task = {
        "Pendulum": "swingup",
        "Cheetah": "run",
        "Humanoid": "walk",
        "Reacher": "easy",
    }
    return dm_make(env_name.lower(), task[env_name], mode=mode)


def visualize():
    """
    :return: visualize a trajectory given the environment.
    """
    pass


def test_env_show():
    env:GraphDmControlWrapper = make('Pendulum', mode='')
    print(env.action_space, env.observation_space)
    env.reset()
    while True:
        img = env.render(mode='rgb_array') # or 'human
        a = env.action_space.sample()
        _, _, done, _ = env.step(a)
        if done:
            break


def test_graph_env():
    env: GraphDmControlWrapper = make('Cheetah', mode='Graph')

    x = env.reset()
    while True:
        a = env.action_space.sample()

        #img = env.render(mode='rgb_array') # or 'human
        #cv2.imwrite('x.jpg', img)
        #cv2.imshow('x.jgp', img)
        #cv2.waitKey(1)
        t, _, done, _ = env.step(a)
        if done:
            break


def test_gnn_model():
    env: GraphDmControlWrapper = make('Cheetah', mode='Graph')

    agent = GNNForwardAgent(0.01, env).cuda()

    s, a, t = [], [], []
    x = env.reset()
    for _ in tqdm.trange(128):
        _a = env.action_space.sample()
        _t, _, _, _ = env.step(_a)
        s.append(x)
        a.append(_a)
        t.append(_t)
        x = _t
    s = togpu(np.array(s))
    a = togpu(np.array(a))
    t = togpu(np.array(t))
    for _ in tqdm.trange(10000):
        agent.update(s, a, t)


def test_inverse_kinematics():
    env: GraphDmControlWrapper = make('Cheetah', mode='Graph')

    for i in range(100):
        a = env.action_space.sample()
        t = env.step(a)[0]

    s, q = env.state_format.decode(t)
    s2, q2 = env.state_format.decode(env.forward(q))
    assert np.abs(q2 - q).sum() < 1e-12

    env.reset()

    qpos, err, iters, suc = env.inv_kinematics(s, u0=q, max_steps=100)[0:4]
    print(q[:env.dq_pos])
    print(qpos)
    print(err, iters, suc)

    xx = env.forward(qpos)
    print((((env.forward(q[:env.dq_pos]) - env.forward(qpos))/np.abs(xx+1e-15))).max())
    exit(0)


def test_reset_geom1():
    import cv2
    env: GraphDmControlWrapper = make('Pendulum', mode='Graph')
    env.reset()
    img = env.render(mode='rgb_array')
    cv2.imwrite('x.jpg', img)
    cv2.waitKey(0)

    print(env.dmcenv.physics.data.xpos)
    env.dmcenv.physics.data.xpos[1] = [3, 0, 0 ]
    env.dmcenv.physics.data.geom_xpos[1] = [3, 0, 0 ]
    #print(env.dmcenv.physics.data.xpos)
    #env.dmcenv.physics.forward()
    print(env.dmcenv.physics.data.geom_xpos)
    img2 = env.render(mode='rgb_array')
    print(env.dmcenv.physics.data.xpos)
    print(env.dmcenv.physics.data.geom_xpos)
    cv2.imshow('y.jpg', np.concatenate((img, img2), axis=1))
    cv2.waitKey(0)


def test_geom():
    import cv2
    env: GraphDmControlWrapper = make('Cheetah', mode='Graph')
    s, q = env.state_format.decode(env.reset())

    g_pos, g_mat = env.recompute_geom(s)

    assert ((g_pos - env.dmcenv.physics.data.geom_xpos)**2).sum() < 1e-10
    a = g_mat
    b = env.dmcenv.physics.data.geom_xmat
    bb = b.copy()
    bb[:, 3:] *= -1
    d = np.minimum(((a-b)**2).sum(axis=1), ((a-bb)**2).sum(axis=1))
    assert d.sum() < 1e-10

def test_render():
    import cv2
    env: GraphDmControlWrapper = make('Cheetah', mode='Graph')

    while True:
        s, q = env.state_format.decode(env.reset())
        img = env.render(mode='rgb_array')

        env.reset()
        for i in range(10):
            a = env.action_space.sample()
            env.step(a)
        img2 = env.render(mode='rgb_array')

        img3 = env.render_state(s)
        cv2.imshow('x', np.concatenate((img, img2, img3), axis=1))
        cv2.waitKey(0)



def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    env: GraphDmControlWrapper = make('Pendulum', mode='Graph')

    path = '/tmp/xxx'
    controller = None
    agent = GNNForwardAgent(0.01, env).cuda()

    model = MBController(agent, controller, 100,
                         init_buffer_size=1000, init_train_step=1000000,
                         valid_ratio=0.2, episode=20, valid_batch_num=3, cache_path=path,
                         vis=Visualizer(os.path.join(path, 'history')))
    model.init(env)



if __name__ == '__main__':
    #main()
    #test_vis()
    #test_geom()
    test_render()
