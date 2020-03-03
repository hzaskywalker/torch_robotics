# here we define the action set of the robot arm
from typing import List
import numpy as np
from transforms3d import quaternions
from robot.envs.hyrule.simulator import Simulator
from sapien.core import Pose
from collections import OrderedDict


def calc_dist(A, B):
    xyz_dist = np.linalg.norm(A.p - B.p)
    theta = quaternions.quat2axangle(quaternions.qmult(A.q, quaternions.qinverse(B.q)))[1]
    return xyz_dist, theta


class Waypoint:
    def __init__(self, agent):
        self.agent = agent

    def arm_contact_cost(self, sim, epsilon=0.01, allowed=None):
        assert  epsilon > 0
        ans = 0
        for i in sim.scene.get_contacts():
            is_arm1 = 'right_' in i.actor1.name
            is_arm2 = 'right_' in i.actor2.name
            if (not is_arm1 and not is_arm2) or (is_arm1 and is_arm2):
                continue
            if allowed is not None:
                if allowed == i.actor1.name or allowed == i.actor2.name:
                    continue
            ans += max(epsilon-i.separation, 0)/epsilon # if i.separation < epsilon, linear, otherwise, 0. When i.separation is 0, it's one
        return ans

    def cost(self, sim):
        raise NotImplementedError


class ArmMove(Waypoint):
    # move to desired position without contact with others
    def __init__(self, agent, target_pose, target_obj=None, contact_epsilon=0.01,
                 weight_xyz=1., weight_angle=0., weight_contact=1.):
        super(ArmMove, self).__init__(agent)

        self.target_pose_p = np.array(target_pose.p)
        self.target_pose_q = np.array(target_pose.q)
        self.target_obj = target_obj
        self.contact_epsilon = contact_epsilon

        self.weight_xyz = weight_xyz
        self.weight_angle = weight_angle
        self.weight_contact = weight_contact


    def cost(self, sim: Simulator):
        agent = sim.objects[self.agent]
        ee_idx = sim._ee_link_idx[self.agent]
        ee_pose = agent.get_links()[ee_idx].pose

        if self.target_obj is None:
            xyz, theta = calc_dist(ee_pose, Pose(self.target_pose_p, self.target_pose_q))
        else:
            raise NotImplementedError

        cost = 0
        if self.weight_contact > 0:
            cost = self.arm_contact_cost(sim, self.contact_epsilon) * self.weight_contact
        return cost + xyz * self.weight_xyz + theta * self.weight_angle


class Grasped(Waypoint):
    def __init__(self, agent, obj_name, weight):
        super(Grasped, self).__init__(agent)
        self.obj_name = obj_name
        self.weight = weight

    def cost(self, sim: Simulator):
        agent = sim.objects[self.agent]

        def is_finger(actor):
            if actor.name == 'right_gripper_finger1_finger_tip_link':
                return 1
            if actor.name == 'right_gripper_finger2_finger_tip_link':
                return 2
            if actor.name == 'right_gripper_finger3_finger_tip_link':
                return 3
            return 0

        mm = np.zeros(3) + np.inf
        object = sim.objects[self.obj_name]
        cmass = object.pose * object.cmass_local_pose
        for link in agent.get_links():
            finger_id = is_finger(link)
            if finger_id:
                link_cmass = link.pose * link.cmass_local_pose
                mm[finger_id-1] = min(mm[finger_id-1], np.linalg.norm(cmass.p - link_cmass.p))
        return mm.sum() * self.weight # 中心的位置。。

    @classmethod
    def load(cls, params: OrderedDict):
        return cls(params['agent'], params['object'], params['weight'])


class ObjectMove(Waypoint):
    def __init__(self, agent, target_pose, weight_xyz=1., weight_angle=0.):
        super(ObjectMove, self).__init__(agent)
        self.target_pose_p = target_pose.p
        self.target_pose_q = target_pose.q

        self.weight_xyz = weight_xyz
        self.weight_angle = weight_angle

    def cost(self, sim: Simulator):
        object = sim.objects[self.agent]
        xyz, theta = calc_dist(object.pose, Pose(self.target_pose_p, self.target_pose_q))
        return xyz * self.weight_xyz + theta * self.weight_angle

    @classmethod
    def load(cls, params: OrderedDict):
        return cls(params['name'], params['target'], params['weight_xyz'], params['weigth_angle'])


class ControlNorm(Waypoint):
    def __init__(self, agent, weight=0.01):
        super(ControlNorm, self).__init__(agent)
        self.weight = weight

    def cost(self, sim: Simulator):
        return (sim.objects[self.agent].get_qf()**2).sum() * self.weight

    @classmethod
    def load(cls, params: OrderedDict):
        return cls(params['name'], params['weight'])


WAYPOINTS = OrderedDict(
    GRASP = Grasped,
    MOVEOBJ = ObjectMove,
    CTRLNORM = ControlNorm
)

class WaypointList(Waypoint):
    def __init__(self, *args):
        object.__init__(self)
        self.args: List[Waypoint] = args

    def cost(self, sim: Simulator):
        total = 0
        for i in self.args:
            total += i.cost(sim)
        return total

    @classmethod
    def load(cls, params: List):
        # TODO: seems very stupid
        return WaypointList( [WAYPOINTS[waypoint[0]].load(waypoint[1]) for waypoint in params])


class Trajectory(Waypoint):
    def __init__(self, *waypoints):
        object.__init__(self)
        assert len(waypoints) > 0, "You must provide more than 1 way point.."
        self.waypoints = waypoints

    def cost(self, sim:Simulator):
        cur = 0
        for cost, t in self.waypoints:
            if cur + t > sim.timestep:
                break
        return cost.cost(sim)

    @classmethod
    def load(cls, params: List):
        return Trajectory([(WaypointList.load(i['list']), i['duration']) for i in params])


def load_waypoints(params):
    if isinstance(params, list):
        # a set of waypoints...
        pass
