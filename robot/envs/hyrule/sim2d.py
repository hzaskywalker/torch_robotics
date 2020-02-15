# A simple game to manipulate the ball..

# first we need to make the simulator and the world body (which is the register
# we need define
import numpy as np
from .simulator import Simulator, Pose, PxIdentity, x2y, x2z, add_link, sapien_core
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
        self._renderer.set_camera_rotation(np.pi/2, -1.5)

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

        ball = self.add_sphere(0, 2, 0.5, ball_color, 'ball', False)
        self.agent = self.add_agent(0, 0, 0.5, agent_color, [[-3, 3], [-3, 3]], 'agent')

    def add_box(self, x, y, size, color, name, fix=False):
        if isinstance(size, int) or isinstance(size, float):
            size = np.array([size, size, size])
        actor_builder = self.scene.create_actor_builder()
        actor_builder.add_box_visual(Pose(), size, color, name)
        actor_builder.add_box_shape(Pose(), size, density=1000)
        box = actor_builder.build(fix)

        pos = Pose(np.array((x, y, size[2]+1e-5)))
        box.set_pose(pos)
        self.objects[name] = box
        return box

    def add_sphere(self, x, y, radius, color, name, fix=False):
        actor_builder = self.scene.create_actor_builder()
        actor_builder.add_sphere_visual(Pose(), radius, color, name)
        actor_builder.add_sphere_shape(Pose(), radius, density=1000)
        box = actor_builder.build(fix)

        pos = Pose(np.array((x, y, radius+1e-5)))
        box.set_pose(pos)
        self.objects[name] = box
        return box

    def add_agent(self, x, y, radius, color, range, name):
        articulation_builder = self.scene.create_articulation_builder()
        world = add_link(articulation_builder, None,  Pose(np.array([x, y, radius + 1e-5]), PxIdentity), name="world")
        yaxis = add_link(articulation_builder, world, ([0, 0., 0], PxIdentity), ((0, 0, 0), PxIdentity), "x",
                         "xaxis", range[0], damping=0.1, type='slider')
        xaxis = add_link(articulation_builder, yaxis, ([0, 0, 0], PxIdentity), ((0, 0, 0), x2y), "y",
                         "yaxis", range[1], damping=0.1, type='slider')

        x2z = np.array([0.7071068, 0, -0.7071068, 0])
        rot = add_link(articulation_builder, xaxis, ([0, 0, 0], PxIdentity), ((0, 0, 0), x2z), "rot", "rot",
                       [-np.pi, np.pi], type='hinge')

        rot.add_sphere_visual(Pose(), radius, color, name)
        rot.add_sphere_visual(Pose((0.3, -0.2, 0.3)), radius * 0.1, (1, 1, 0), name)
        rot.add_sphere_visual(Pose((0.3, 0.2, 0.3)), radius * 0.1, (1, 1, 0), name)
        rot.add_sphere_shape(Pose(), radius, density=1000)
        wrapper = articulation_builder.build(True) #fix base = True
        self.objects[name] = wrapper
        return wrapper

    def get_xy(self, name):
        if isinstance(self.objects, sapien_core.pysapien.Articulation):
            raise NotImplementedError
        return self.objects[name].pose.p[:2]

    def set_xy(self, name, x, y):
        if isinstance(self.objects, sapien_core.pysapien.Articulation):
            raise NotImplementedError
        pose =  self.objects[name].pose
        self.objects[name].set_pose(Pose((x, y, pose.p[2]), pose.q))

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
        agent = simulator.agent
        pose = agent.get_qpos()
        x, y, theta = pose
        x = np.round(x + np.cos(theta))
        y = np.round(y + np.sin(theta))
        if self.moveable(simulator, x, y):
            agent.set_qpos([x, y, theta])

class Rot(Magic):
    #ROT_LEFT = np.array()
    def __init__(self, dir):
        self.dir = dir
    def forward(self, simulator, step):
        if step != 0:
            return
        agent = simulator.agent
        x, y, theta = agent.get_qpos()
        theta += self.dir * np.pi/2
        while theta >= np.pi:
            theta -= 2 * np.pi
        while theta < -np.pi:
            theta += 2 * np.pi
        agent.set_qpos([x, y, theta])

class MagicalTelepotation(Move):
    def __init__(self, ball_name):
        self.ball_name = ball_name

    def forward(self, simulator: Sim2D, step):
        if step == -1:
            assert self.ball_name in simulator.objects, "argument of MagicalTelepotation should be the name of a actor"
            # will be done after each timestep..
            x, y, theta = simulator.agent.get_qpos()

            x = np.round(x + np.cos(theta))
            y = np.round(y + np.sin(theta))
            simulator.set_xy(self.ball_name, x, y)

class Round(Magic):
    # TODO: set all coordinates into int......
    def forward(self, simulator, step):
        if step == -1:
            raise NotImplementedError

class ControlPanelV1(ControlPanel):
    def __init__(self, sim):
        super(ControlPanelV1, self).__init__(sim)
        self.register('move', Move)
        self.register('rot', Rot, 'dir')
        self.register('transport', MagicalTelepotation)


# define relation ... just a wrapper of instruction ...

# define the high-level language ...

# and a example


