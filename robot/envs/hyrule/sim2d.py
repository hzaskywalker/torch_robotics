# A simple game to manipulate the ball..

# first we need to make the simulator and the world body (which is the register
# we need define
import numpy as np
import logging
from .gameplay.simulator import Simulator, Pose, PxIdentity, x2y, add_link, sapien_core
from .gameplay import Magic, World, Object, Action, Constraint, Relation

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
        wall_color = (0, 0, 0)
        agent_color = (0, 1, 0)
        ball_color = (0, 0, 1)

        self.scene.add_ground(0.)
        self.add_box(-3.5, 0, (0.5, 3, 0.5), wall_color, '_box1', True)
        self.add_box(+3.5, 0, (0.5, 3, 0.5), wall_color, 'box1', True)
        self.add_box(0, +3.5, (4, 0.5, 0.5), wall_color, 'box1', True)
        self.add_box(0, -3.5, (4, 0.5, 0.5), wall_color, 'box1', True)

        ball = self.add_sphere(0, 2, 0.5, ball_color, 'ball', True)
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
        box.set_name(name)
        self.objects[name] = box
        return box

    def add_sphere(self, x, y, radius, color, name, fix=False):
        actor_builder = self.scene.create_actor_builder()
        actor_builder.add_sphere_visual(Pose(), radius, color, name)
        actor_builder.add_sphere_shape(Pose(), radius, density=1000)
        box = actor_builder.build(fix)

        pos = Pose(np.array((x, y, radius+1e-5)))
        box.set_pose(pos)
        box.set_name(name)
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
        wrapper.set_name(name)
        self.objects[name] = wrapper
        return wrapper

def get_xy(obj):
    if isinstance(obj, sapien_core.pysapien.Articulation):
        raise NotImplementedError
    return obj.pose.p[:2]

def set_xy(obj, x, y):
    if isinstance(obj, sapien_core.pysapien.Articulation):
        raise NotImplementedError
    pose = obj.pose
    obj.set_pose(Pose((x, y, pose.p[2]), pose.q))

def move_agent(agent):
    pose = agent.get_qpos()
    x, y, theta = pose
    x = np.round(x + np.cos(theta))
    y = np.round(y + np.sin(theta))
    return x, y, theta


def moveable(simulator, x, y, obstacles=('ball',)):
    for name in obstacles:
        px, py = get_xy(simulator.objects[name])
        if ((px - x) ** 2 + (py-y) ** 2) < 1:
            return False
    if abs(x)+0.5 >= 3 or abs(y)+0.5 >= 3:
        return False
    return True
# define instructions by implement a control panel...
# what's the best form of instructions? string, or class?

class Move(Magic):
    def __init__(self, agent=None):
        self.agent = agent
        Magic.__init__(self)

    def forward(self, simulator):
        agent = self.agent if self.agent is not None else simulator.agent
        x, y, theta = move_agent(agent)
        agent.set_qpos([x, y, theta])
        return 0

class Rot(Magic):
    #ROT_LEFT = np.array()
    def __init__(self, dir, agent=None):
        self.dir = dir
        self.agent = agent
        Magic.__init__(self)

    def forward(self, simulator):
        agent = self.agent if self.agent is not None else simulator.agent
        x, y, theta = agent.get_qpos()
        theta += self.dir * np.pi/2
        while theta >= np.pi:
            theta -= 2 * np.pi
        while theta < -np.pi:
            theta += 2 * np.pi
        agent.set_qpos([x, y, theta])
        return 0

class MagicalTelepotation(Move):
    def __init__(self, actor, agent=None):
        self.actor = actor
        self.agent = agent
        Magic.__init__(self)

    def forward(self, simulator: Sim2D):
        agent = self.agent if self.agent is not None else simulator.agent
        x, y, theta = agent.get_qpos()

        x = np.round(x + np.cos(theta))
        y = np.round(y + np.sin(theta))
        set_xy(self.actor, x, y)
        #TODO: need change some behaviour
        return 0

class Sim2DV1(Sim2D):
    def __init__(self):
        super(Sim2DV1, self).__init__()
        self.register('move', Move)
        self.register('rot', Rot)
        self.register('transport', MagicalTelepotation)

class CollsionMove(Relation):
    def __init__(self, perpectual=False):
        Relation.__init__(self, timestep=1, perpetual=False)

    def execute(self, object: Object, sim: Simulator):
        x, y, theta = move_agent(object.pointer)
        obstacles = [i.pointer.name for i in object.world.objects.values() if i not in object.child and i != object]
        if len(object.child) > 0:
            x = x + np.cos(theta)
            y = y + np.sin(theta)
        if moveable(sim, x, y, obstacles):
            sim.execute('move', object.pointer)
        return 1

    def prerequisites(self, object: Object, parent: Object):
        # you can add moving command at any time ...
        return True

class Grasped(Relation):
    # execution of the constraint
    def __init__(self, perpectual=True):
        Relation.__init__(self, timestep=1, perpetual=1)

    def execute(self, object: Object, sim: Simulator):
        xx, yy, _ = move_agent(object.parent.pointer)
        if moveable(sim, xx, yy, ()):
            sim.execute('transport', object.pointer, object.parent.pointer)
            return 0
        logging.warning("GRASPING FAILED BECAUSE OF OBSTACLES")
        return 1

    def prerequisites(self, object: Object, parent: Object):
        x, y = get_xy(object.pointer)
        px, py, theta = parent.pointer.get_qpos()
        xx = np.round(px + np.cos(theta))
        yy = np.round(py + np.sin(theta))
        return abs(xx-x) + abs(yy-y) < 1e-5



class Sim2DWorld(World):
    def __init__(self, sim):
        objects = {
            'agent': Object(sim.agent, None),
            'ball': Object(sim.ball, None),
        }
        super(Sim2DWorld, self).__init__(sim, objects, None)
        self.register('rot', lambda dir: Action('rot', dir, timestep=0, perpetual=False))
        self.register('transport', lambda: Constraint('transport', timestep=1, perpertual=True))

        self.register('move', CollsionMove)
        self.register('grasped', Grasped)
