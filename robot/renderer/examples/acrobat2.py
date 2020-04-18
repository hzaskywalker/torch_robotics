import numpy as np
import os
import trimesh

def Acrobat2Render(path=None, mode='rgb_array', model=None, env=None, agent=None):
    if mode == 'rgb_array':
        os.environ['PYOPENGL_PLATFORM'] = 'osmesa'  # remove this for human mode ..
    from ..renderer import Renderer

    if path is not None and os.path.exists(path):
        return Renderer.load(path)

    if model is None:
        from robot.model.arm.exp.sapien_validator import get_env_agent, build_diff_model
        env, agent = get_env_agent()
        model = build_diff_model(env)

    from robot import tr
    r = Renderer()
    arm = r.make_arm(model.M, model.A, name='arm')

    for i in range(len(model.M)-1):
        # pass
        length = abs(model.A[i].detach().cpu().numpy()[3]) * 2

        c = r.make_compose()
        capsule = r.make_capsule(length, 0.1, (255, 255, 255), np.eye(4))
        pose =np.array(
            [[1, 0, 0, 0],
             [0, 0, 1, -length/2],
             [0, 1, 0, 0],
             [0, 0, 0, 1]]
        )
        c.add_shapes(capsule, pose)

        sphere = r.make_sphere((0, 0, 0), 0.1, (255, 0, 0))
        pose = np.eye(4)
        pose[1,3] = length/2
        c.add_shapes(sphere, pose)

        arm.add_shapes(c, np.eye(4))

        r.add_point_light([2, 2, 2], [255, 255, 255])
        r.add_point_light([2, -2, 2], [255, 255, 255])
    r.add_point_light([-2, 0, 2], [255, 255, 255])
    r.set_camera_position(0, 0, 3)
    r.set_camera_rotation(1.57, -1.57)

    #print(r._get_camera_pose()@np.array([1, 0, 0, 0]))
    #print(r._get_camera_pose()@np.array([0, 1, 0, 0]))
    #print(r._get_camera_pose()@np.array([0, 0, 1, 0]))
    #exit(0)

    arm.set_pose(np.array((0, 0)))

    #capsule = r.make_capsule(1, 0.1, (0, 0, 255), pose)
    #sphere = r.make_sphere((0, 0 ,0), 0.2, (255, 0, 0))

    if path is not None:
        r.save(path)
    return r
