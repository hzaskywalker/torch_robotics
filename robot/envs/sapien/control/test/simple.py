import sapien.core as sapien_core
import numpy as np
from sapien.core import Pose
from transforms3d.quaternions import rotate_vector, qmult

PxIdentity = np.array([1, 0, 0, 0])
x2y = np.array([0.7071068, 0, 0, 0.7071068])
x2z = np.array([0.7071068, 0, -0.7071068, 0])

def main():
    sim = sapien_core.Simulation()
    renderer = sapien_core.OptifuserRenderer()
    sim.set_renderer(renderer)
    scene = sim.create_scene(gravity=np.array([0., 0., 0.]))

    scene.set_timestep(0.01)

    scene.set_ambient_light([.4, .4, .4])
    scene.set_shadow_light([1, -1, -1], [.5, .5, .5])

    render_controller = sapien_core.OptifuserController(renderer)

    render_controller.set_camera_position(0., -2., 2)
    render_controller.set_camera_rotation(-np.pi - np.pi / 2, -0.9)

    builder = scene.create_articulation_builder()

    world = builder.create_link_builder(None)
    world.set_name('world')

    fake = builder.create_link_builder(world)
    fake.set_name('fake')
    fake.set_joint_name('x')
    fake.set_joint_properties(sapien_core.ArticulationJointType.PRISMATIC, [[-1, 1]])
    #fake.set_mass_and_inertia(1e-8, Pose(), [1, 1, 1])

    object = builder.create_link_builder(fake)
    object.set_name('object')
    object.set_joint_name('y')
    object.set_joint_properties(sapien_core.ArticulationJointType.PRISMATIC, [[-1, 1]], Pose([0, 0, 0], x2y))
    #object.set_mass_and_inertia(1e-6, Pose(), [1e-6, 1e-6, 1e-6])

    # TODO: changing density to 0.001, 1, 100, 10000
    object.add_capsule_shape(Pose([0, 0, 0], x2z), 0.05, 0.05, density=0.0001)  # half length
    object.add_capsule_visual(Pose([0, 0, 0], x2z), 0.05, 0.05, (1, 1, 1), 'object')  # half length

    wrapper = builder.build(True)  # fix base = True
    wrapper.set_root_pose(Pose([0., 0., 0.]))

    render_controller.show_window()
    render_controller.set_current_scene(scene)

    for i in wrapper.get_links():
        if i.name == 'object':
            object = i
        print(i.mass)
    for i in wrapper.get_joints():
        i.set_friction(0)

    for id in range(10000):
        object.add_force_torque([-0.1, -0.1, 0], [0, 0, 0])
        scene.step()
        if id % 5 == 0:
            scene.update_render()
            render_controller.render()
    del scene


if __name__ == '__main__':
    main()
