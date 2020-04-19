import numpy as np
import os
import trimesh

def Acrobat2Render(path=None, model=None, env=None, agent=None):
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
        length = abs(model.A[i].detach().cpu().numpy()[3]) * 2
        capsule = r.capsule(length, 0.1, (255, 255, 255, 127), r.x2y())
        arm.add_shapes(r.compose(capsule, r.axis(np.eye(4), scale=0.2),
                                 r.sphere((0, -length/2, 0), 0.1, (255, 0, 0))))


    r.line([-3, 0, 0], [3, 0, 0], 0.02, (255, 255, 255))

    r.add_point_light([0, 2, 2], [255, 255, 255], intensity=100)
    r.add_point_light([0, -2, 2], [255, 255, 255], intensity=100)
    r.set_camera_position(0, 0, 3)
    r.set_camera_rotation(1.57, -1.57)

    arm.set_pose(np.array((0, 0)))

    if path is not None:
        r.save(path)
    return r
