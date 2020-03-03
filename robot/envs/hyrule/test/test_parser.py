from robot.envs.hyrule import Simulator, load_scene, load_json, dump_json
from collections import OrderedDict

def test():
    params = OrderedDict(
        ground=0,
        agent=OrderedDict(
            type='robot',
            lock=['pan_joint', 'tilt_joint', 'linear_joint'],
            lock_value = [0, 0, 0]
        ),
        table=OrderedDict(
            type='box',
            density=100000,
            fix=True,
        ),
        box=OrderedDict(
            type='box',
            density=1000,
            center=[0.8, 0, 0.5 + 0.02],
            size=[0.02, 0.02, 0.02],
            color=[0, 1, 0],
            fix=False,
        )
    )

    trajectories = [
        [
            ['GRASP', {}]
        ],
    ]

    env = Simulator()
    load_scene(env, params)

    for i in range(10000):
        env.render()
        env.step()

if __name__ == '__main__':
    test()