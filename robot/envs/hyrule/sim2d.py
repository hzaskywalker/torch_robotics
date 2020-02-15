# A simple game to manipulate the ball..

# first we need to make the simulator and the world body (which is the register
# we need define
import numpy as np
from .simulator import Simulator, Pose, PxIdentity, x2y, x2z, add_link
from .gameplay import ControlPanel, Magic

# define the sim
class Sim2D(Simulator):
    def build_renderer(self):
        self.scene.set_ambient_light([.4, .4, .4])
        self.scene.set_shadow_light([1, -1, -1], [.5, .5, .5])
        self.scene.add_point_light([2, 2, 2], [1, 1, 1])
        self.scene.add_point_light([2, -2, 2], [1, 1, 1])
        self.scene.add_point_light([-2, 0, 2], [1, 1, 1])

        self._renderer.set_camera_position(0, 0, 20)
        self._renderer.set_camera_rotation(0, -1.5)

    def build_scene(self):
        self.objects = {}
        wall_color = (0, 0, 0)
        agent_color = (0, 1, 0)
        ball_color = (0, 0, 1)

        self.scene.add_ground(0.)
        self.add_box(-3.5, 0, (0.5, 3, 0.5), wall_color, '_box1', True)
        self.add_box(+3.5, 0, (0.5, 3, 0.5), wall_color, 'box1', True)
        self.add_box(0, +3.5, (4, 0.5, 0.5), wall_color, 'box1', True)
        self.add_box(0, -3.5, (4, 0.5, 0.5), wall_color, 'box1', True)

        self.add_sphere(0, 2, 0.5, ball_color, 'ball', False)
        self.agent = self.add_agent(0, 0, 0.5, agent_color, [[-3, 3], [-3, 3]], 'agent')

    def add_box(self, x, y, size, color, name, fix=False):
        if isinstance(size, int) or isinstance(size, float):
            size = np.array([size, size, size])
        pos = Pose(np.array((x, y, size[2]+1e-5)))
        actor_builder = self.scene.create_actor_builder()
        actor_builder.add_box_visual(pos, size, color, name)
        actor_builder.add_box_shape(pos, size, density=1000)
        box = actor_builder.build(fix)
        self.objects[name] = box
        return box

    def add_sphere(self, x, y, radius, color, name, fix=False):
        pos = Pose(np.array((x, y, radius+1e-5)))
        actor_builder = self.scene.create_actor_builder()
        actor_builder.add_sphere_visual(pos, radius, color, name)
        actor_builder.add_sphere_shape(pos, radius, density=1000)
        box = actor_builder.build(fix)
        self.objects[name] = box
        return box

    def add_agent(self, x, y, radius, color, range, name):
        articulation_builder = self.scene.create_articulation_builder()
        world = add_link(articulation_builder, None,  Pose(np.array([x, y, radius + 1e-5]), PxIdentity), name="world")
        yaxis = add_link(articulation_builder, world, ([0, 0., 0], PxIdentity), ((0, 0, 0), x2y), "y",
                                      "yaxis", range[0], damping=0.1, type='slider')
        xaxis = add_link(articulation_builder, yaxis, ([0, 0, 0], PxIdentity), ((0, 0, 0), PxIdentity), "x",
                                      "xaxis", range[1], damping=0.1, type='slider')
        rot = add_link(articulation_builder, xaxis, ([0, 0, 0], PxIdentity), ((0, 0, 0), x2z), "x",
                         "rot", [-np.pi, np.pi], type='hinge')

        rot.add_sphere_visual(Pose(), radius, color, name)
        rot.add_sphere_visual(Pose((0.3, -0.2, 0.3)), radius * 0.1, (1, 1, 0), name)
        rot.add_sphere_visual(Pose((0.3, 0.2, 0.3)), radius * 0.1, (1, 1, 0), name)
        rot.add_sphere_shape(Pose(), radius, density=1000)
        wrapper = articulation_builder.build(True) #fix base = True
        self.objects[name] = wrapper
        return wrapper

    def get_xy(self, name):
        return self.objects[name].pose.p[:2]

    def set_xy(self, name, x, y):
        p = self.objects[name].pose.p
        self.objects[name].set_position((x, y, p[2]))

# define instructions by implement a control panel...
# what's the best form of instructions? string, or class?


class Move(Magic):
    def moveable(self, simulator, x, y):
        for name in ['ball']:
            px, py = simulator.get_xy(name)
            if ((px - x) ** 2 + (py-y) ** 2) < 1:
                return False
        if abs(x) > 3 or abs(y) > 3:
            return False
        return True

    def forward(self, simulator, step):
        if step != 0:
            return
        pose = simulator.objects['agent'].pose
        x, y = pose.p[:2]
        from transforms3d.quaternions import rotate_vector
        dx, dy = rotate_vector((1, 0, 0), pose.q)[:2]
        x = np.round(x + dx)
        y = np.round(y + dy)
        if self.moveable(simulator, x, y):
            simulator.set_xy('agent', x, y)

class Rot(Magic):
    #ROT_LEFT = np.array()
    def __init__(self, dir):
        pass


class ControlPanelV1(ControlPanel):
    def __init__(self, sim):
        super(ControlPanelV1, self).__init__(sim)
        self.register('move', Move)

# define relation ... just a wrapper of instruction ...

# define the high-level language ...

# and a example


