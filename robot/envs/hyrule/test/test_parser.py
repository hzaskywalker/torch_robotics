from robot.envs.hyrule import Simulator, load_scene, load_json, dump_json, load_waypoints
from collections import OrderedDict

def test():
    params = OrderedDict(
        ground=0,
        agent=OrderedDict(
            type='robot',
            lock=['pan_joint', 'tilt_joint', 'linear_joint'],
            lock_value = [0, 0, 0],
            actuator=['right_shoulder_pan_joint',
                      'right_shoulder_lift_joint',
                      'right_arm_half_joint',
                      'right_elbow_joint',
                      'right_wrist_spherical_1_joint',
                      'right_wrist_spherical_2_joint',
                      'right_wrist_3_joint',
                      'right_gripper_finger3_joint',
                      'right_gripper_finger2_joint',
                      'right_gripper_finger1_joint'],
            actuator_range = [[-50, 50] for i in range(7)] + [[-5, 5] for i in range(3)]
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

    waypoints = [
        OrderedDict(
            list=[
                ['GRASP', dict(agent='agent', object='box', weight=1)],
                ['MOVEOBJ', dict(name='box', target=[0.9, 0.2, 1.], weight_xyz=2, weight_angle=0)],
                ['CTRLNORM', dict(name='agent', weight=0.0001)]
            ],
            duration=50,
        ),
        OrderedDict(
            list=[
                ['GRASP', dict(agent='agent', object='box', weight=1)],
                ['MOVEOBJ', dict(name='box', target=[0.8, -0.2, 0.55], weight_xyz=2, weight_angle=0)],
                ['CTRLNORM', dict(name='agent', weight=0.0001)],
            ],
            duration = 50,
        )
    ]

    params['waypoints'] = waypoints

    env = Simulator()
    load_scene(env, params)

    env.reset()
    #print(env.action_space, env.observation_space)
    for i in range(10000):
        env.render()
        action = env.action_space.sample() * 0
        env.step(action)
        if i % 100 == 0:
            env.reset()

if __name__ == '__main__':
    test()