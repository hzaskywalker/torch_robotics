import numpy as np
import os
import trimesh

def ArmreachRenderer(path=None, model=None, env=None, agent=None):
    from ..renderer import Renderer

    if path is not None and os.path.exists(path):
        return Renderer.load(path)

    if env is None:
        from robot.model.arm.exp.arm_validator import get_env_agent, build_diff_model
        env, agent = get_env_agent()
        model = build_diff_model(env)

    from robot import renderer, tr
    r = renderer.Renderer()
    arm = r.make_arm(model.M, model.A, name='arm')

    q = agent.get_qpos()
    agent.set_qpos(q * 0)
    def get_mesh(i):
        shape = i.get_collision_shapes()
        if len(shape)==0:
            return None, None
        shape = shape[0]
        mesh = shape.convex_mesh_geometry
        # local_pose = np.eye(4) # TODO: change it to the local of the first link instead of the CMASS of the gripper.
        tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.indices.reshape(-1, 3),
                             face_colors=list(np.random.randint(255 - 32, size=(3,)) + 32)+[48])
        return tm, shape.pose

    for idx, i in enumerate(agent.get_links()[1:8]):
        if idx < 6:
            tm, local_pose = get_mesh(i)
            local_pose = tr.pose2SE3(i.cmass_local_pose.inv() * local_pose)
            arm.add_shapes(r.trimesh(tm, pose=local_pose))
        else:
            # TODO: there is no shape for the last link in the kinova robot
            T = arm.fk(q*0)[-2:-1] #
            compose = r.compose()
            for l in i.links:
                tm, local_pose = get_mesh(l)
                if tm is None: continue
                space_pose = tr.totensor(tr.pose2SE3(l.get_pose() * local_pose))
                local_pose = tr.tocpu(tr.inv_trans(tr.totensor(T)) @ space_pose[None,:])[0]
                compose.add_shapes(r.trimesh(tm, pose=local_pose))
            arm.add_shapes(compose)

    ee_sphere = r.sphere((0, 0, 0), 0.01, color=(255, 0, 0))
    arm.add_shapes(ee_sphere)

    r.add_point_light([2, 2, 2], [255, 255, 255])
    r.add_point_light([2, -2, 2], [255, 255, 255])
    r.add_point_light([-2, 0, 2], [255, 255, 255])

    r.set_camera_position(1.2, -0.5, 1.2)
    r.set_camera_rotation(-3.14 - 0.5, -0.2)

    if path is not None:
        r.save(path)
    return r
