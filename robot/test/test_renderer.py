import tqdm
import numpy as np
import trimesh
import pickle
import cv2

def test_sphere():
    from robot import renderer
    r = renderer.Renderer()

    sphere = r.make_sphere((0, 0, 0), 3, (255, 255, 255))
    r.add_point_light((0, 5, 0), color=(255, 0, 0))
    r.set_camera_position(0, 0, 10)

    for i in tqdm.trange(1000):
        pos = np.random.random((3,))
        sphere.set_center(pos)
        img = r.render()


def test_arm():
    from robot.model.arm.exp.arm_validator import get_env_agent, build_diff_model
    env, agent = get_env_agent()
    model = build_diff_model(env)
    np.random.seed(3)
    q = np.random.random((7,))* np.pi * 2
    agent.set_qpos(q)
    for i in agent.get_links():
        print(i.get_pose())
    img2 = env.render(mode='rgb_array')

    from robot import renderer, tr
    r = renderer.Renderer()
    arm = r.make_arm(model.M, model.A)

    agent.set_qpos(q * 0)
    def get_mesh(i):
        shape = i.get_collision_shapes()
        if len(shape)==0:
            return None, None
        shape = shape[0]
        mesh = shape.convex_mesh_geometry
        # local_pose = np.eye(4) # TODO: change it to the local of the first link instead of the CMASS of the gripper.
        tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.indices.reshape(-1, 3),
                             face_colors=np.random.randint(255 - 32, size=(3,)) + 32)
        return tm, shape.pose

    for idx, i in enumerate(agent.get_links()[1:8]):
        if idx <6:
            tm, local_pose = get_mesh(i)
            local_pose = tr.pose2SE3(i.cmass_local_pose.inv() * local_pose)
            arm.add_shapes(r.make_mesh(tm), local_pose)
        else:
            # TODO: there is no shape for the last link in the kinova robot
            T = arm.fk(q*0)[-2:-1] #
            compose = r.make_compose()
            for l in i.links:
                tm, local_pose = get_mesh(l)
                if tm is None: continue
                space_pose = tr.togpu(tr.pose2SE3(l.get_pose() * local_pose))
                local_pose = tr.tocpu(tr.inv_trans(T) @ space_pose[None,:])[0]
                compose.add_shapes(r.make_mesh(tm), local_pose)
            arm.add_shapes(compose, np.eye(4))

    #print(agent.get_ee_links().get_collision_shapes())
    #exit(0)
    ee_sphere = r.make_sphere((0, 0, 0), 0.01, color=(255, 0, 0))
    arm.add_shapes(ee_sphere, local_pose=np.eye(4))
    arm.set_pose(q)

    r.add_point_light([2, 2, 2], [255, 255, 255])
    r.add_point_light([2, -2, 2], [255, 255, 255])
    r.add_point_light([-2, 0, 2], [255, 255, 255])
    r.set_camera_position(1.2, -0.5, 1.2)
    r.set_camera_rotation(-3.14 - 0.5, -0.2)

    img = r.render()
    cv2.imwrite('x.jpg', np.concatenate((img, img2), axis=1))

if __name__:
    #test_sphere()
    test_arm()