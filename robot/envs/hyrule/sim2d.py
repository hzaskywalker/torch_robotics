# A simple game to manipulate the ball..

# first we need to make the simulator and the world body (which is the register
# we need define
import numpy as np
from .gameplay.simulator import Simulator, Pose, PxIdentity, x2y, add_link, sapien_core
from .gameplay import Constraint

# define the sim
class Sim2D(Simulator):
    def __init__(self):
        super(Sim2D, self).__init__()
        self.register('move', MoveAgent)
        self.register('rot', RotAgent)
        self.register('fixed', Fixed)
        self.register('nocollision', NoConllision)
        self.nocollision()

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


def move_agent(agent):
    pose = agent.get_qpos()
    x, y, theta = pose
    x = np.round(x + np.cos(theta))
    y = np.round(y + np.sin(theta))
    return x, y, theta


class NoConllision(Constraint):
    def __init__(self):
        Constraint.__init__(self, priority=9, perpetual=True)

    def check_wall(self, x, y):
        if abs(x)+0.5 >= 3 or abs(y)+0.5 >= 3:
            return False
        return True

    def cost(self, sim_t, s, t):
        bx, by = t['ball']['pose'].p[0:2]
        ax, ay = t['agent']['qpos'][0:2]
        return 1-(self.check_wall(ax, ay) and self.check_wall(bx, by) and np.abs(bx - ax)**2 + np.abs(by-ay)**2 >= 1 - 1e-5)


class MoveAgent(Constraint):
    def __init__(self, agent):
        self.agent = agent
        self.name = self.agent.name
        Constraint.__init__(self, priority=1, perpetual=0)

    def prerequisites(self, sim):
        return True

    def preprocess(self, sim):
        x, y, theta = move_agent(self.agent)
        self.agent.set_qpos([x, y, theta])

    def cost(self, sim_t, s, t):
        #actually because it has very low priority, we don't care if it violates the constraint...
        s_qpos = s[self.name]['qpos']
        t_qpos = t[self.name]['qpos']
        diff = t_qpos[:2] - s_qpos[:2]
        target = [np.cos(s_qpos[-1]), np.sin(s_qpos[-1])]
        return 1-(np.abs(diff - np.array(target)).sum() < 1e-5)


class RotAgent(Constraint):
    def __init__(self, agent, dir):
        self.agent = agent
        self.dir = dir
        self.name = self.agent.name
        super(RotAgent, self).__init__(priority=0, perpetual=False)

    def rot(self, x, y, theta):
        theta += self.dir * np.pi/2
        while theta >= np.pi:
            theta -= 2 * np.pi
        while theta < -np.pi:
            theta += 2 * np.pi
        return theta

    def postprocess(self, sim):
        x, y, theta = self.agent.get_qpos()
        theta = self.rot(x, y, theta)
        self.agent.set_qpos([x, y, theta])

    def cost(self, sim_t, s, t):
        #actually because it has very low priority, we don't care if it violates the constraint...
        dist = self.rot(*s[self.name]['qpos']) - t[self.name]['qpos'][2]
        return 1-(np.abs(dist) < 1e-5)


class Fixed(Constraint):
    def __init__(self, actor, agent):
        self.actor = actor
        self.agent = agent
        super(Fixed, self).__init__(5, perpetual=True)

    def postprocess(self, simulator: Sim2D):
        x, y, theta = move_agent(self.agent)

        pose = self.actor.pose
        self.actor.set_pose(Pose((x, y, pose.p[2]), pose.q))

    def check(self, sim):
        bx, by = sim.ball.pose.p[:2]
        ax, ay, theta = sim.agent.get_qpos()
        return ((np.array([bx-ax, by-ay]) - np.array([np.cos(theta), np.sin(theta)]))**2).sum() < 1e-5

    def cost(self, sim_t, s, t):
        return 1-self.check(sim_t)

    def prerequisites(self, sim):
        return self.check(sim)
