import numpy as np
import cv2
from robot.envs.dm_control import make as dm_make
from robot.envs.dm_control.graph_env import GraphDmControlWrapper

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


def test_env_show():
    env:GraphDmControlWrapper = make('Pendulum', mode='')
    print(env.action_space, env.observation_space)
    env.reset()
    while True:
        img = env.render(mode='rgb_array') # or 'human
        cv2.imshow('x', img)
        cv2.waitKey(1)
        a = env.action_space.sample()
        _, _, done, _ = env.step(a)
        if done:
            break




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
    s, q = env.state_prior.decode(env.reset())

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
        s, q = env.state_prior.decode(env.reset())
        img = env.render(mode='rgb_array')

        env.reset()
        for i in range(10):
            a = env.action_space.sample()
            env.step(a)
        img2 = env.render(mode='rgb_array')

        img3 = env.render_state(s)
        cv2.imshow('x', np.concatenate((img, img2, img3), axis=1))
        cv2.waitKey(0)


def test_graph_env():
    env: GraphDmControlWrapper = make('Cheetah', mode='Graph')

    x = env.reset()
    while True:
        a = env.action_space.sample()

        img = env.render(mode='rgb_array') # or 'human
        #cv2.imwrite('x.jpg', img)
        cv2.imshow('x.jgp', img)
        cv2.waitKey(1)
        t, _, done, _ = env.step(a)
        if done:
            break


def test_xx():
    env: GraphDmControlWrapper = make('Cheetah', mode='Graph')

    state_space = env.observation_space
    action_space = env.action_space
    global_space = env.global_space

    scene = env.get_global()

    state = env.reset()

    frame = state_space(state, is_batch=False, scene=scene)
    assert state in state_space

    a = action_space.sample()
    assert a in action_space

    assert action_space.observe(a).shape == action_space.observation_shape

    state2 = env.step(a)[0]
    assert state2 in state_space

    w_space = state_space['w']
    h_space = state_space['p']

    # derivate space
    assert abs(state_space.metric(state_space.sub(state, state))) < 1e-5
    derivative = state_space.sub(state2, state)
    state3 = state_space.add(derivative, state)
    assert abs(state_space.metric(state_space.sub(state2, state3))) < 1e-5



if __name__ == '__main__':
    #test_graph_env()
    test_xx()
