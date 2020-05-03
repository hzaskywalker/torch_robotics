import sapien.core as sapien_core
import numpy as np
import argparse

from sapien.core import Pose

def test_newton_ball2():
    engine = sapien_core.Engine()

    renderer = sapien_core.OptifuserRenderer()
    engine.set_renderer(renderer)

    scene = engine.create_scene(gravity=np.array([0, 0, 0]))
    dt = 0.00001
    scene.set_timestep(dt)
    material = engine.create_physical_material(0, 0, 1)

    balls = []
    for i in range(3):
        actor = scene.create_actor_builder()
        actor.add_sphere_shape(radius=0.05, material=material, density=1000)
        actor.add_sphere_visual(radius=0.05, color=np.array((0.8, 0.8, 1)))
        balls.append(actor.build())

        balls[-1].set_pose(Pose((5 + i * 0.12, 0, 0.05)))
        balls[-1].set_velocity(np.array([0, 0, 0]))
        balls[-1].set_angular_velocity(np.array([0, 0, 0]))

    scene.add_ground(0, material=material)

    balls[0].set_pose(Pose((5 - 1, 0, 0.05)))
    balls[0].set_velocity(np.array([1, 0, 0]))

    render_controller = sapien_core.OptifuserController(renderer)
    render_controller.set_current_scene(scene)

    scene.set_ambient_light([.4, .4, .4])
    scene.set_shadow_light([1, -1, -1], [.5, .5, .5])
    render_controller.set_camera_position(5, -3, 2)
    render_controller.set_camera_rotation(1.57, -0.5)
    render_controller.show_window()

    while True:
        for i in range(int(0.02/dt)):
            scene.step()
        scene.update_render()
        render_controller.render()
        print(balls[0].get_pose())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    choice = [i[5:] for i in globals() if i.startswith('test_')]
    parser.add_argument('task', type=str, choices=choice)
    args = parser.parse_args()
    eval('test_'+args.task+'()')
