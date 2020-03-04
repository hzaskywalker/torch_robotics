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


class Waypoint:
    def __init__(self, agent):
        self.agent = agent
        self._goal_dim = 0

    def cost(self, sim):
        return self.compute_cost(*self._get_obs(sim), None)

    def _get_obs(self, sim: Simulator):
        return None, None

    def compute_cost(self, achieved, target, info):
        raise NotImplementedError


class ArmMove(Waypoint):
    # move to desired position without contact with others
    def __init__(self, agent, target, weight=1.):
        super(ArmMove, self).__init__(agent)

        self.target = np.array(target)
        self.weight = weight
        self._goal_dim = 3

    @classmethod
    def load(cls, params: OrderedDict):
        return cls(params['agent'], params['target'], params['weight'])

    def _get_obs(self, sim: Simulator):
        agent = sim.objects[self.agent]
        ee_idx = sim._ee_link_idx[self.agent]
        ee_pose = agent.get_links()[ee_idx].pose
        return ee_pose.p, self.target

    def compute_cost(self, achieved, target, info):
        return self.weight * np.linalg.norm(achieved - target)


class Grasped(Waypoint):
    def __init__(self, agent, obj_name, weight):
        super(Grasped, self).__init__(agent)
        self.obj_name = obj_name
        self.weight = weight

        self._goal_dim = 1

    @classmethod
    def load(cls, params: OrderedDict):
        return cls(params['agent'], params['object'], params['weight'])

    def _get_obs(self, sim: Simulator):
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
        return (mm.sum(),), (0,)

    def compute_cost(self, achieved, target, info):
        achieved = np.array(achieved)
        if len(achieved.shape) == 1: achieved = achieved[0]
        else: achieved = achieved[..., 0]
        return achieved.clip(0, 3) * self.weight


class ObjectMove(Waypoint):
    def __init__(self, agent, target_pose, weight_xyz=1., weight_angle=0.):
        super(ObjectMove, self).__init__(agent)
        self.target_pose_p = target_pose.p
        self.target_pose_q = target_pose.q

        self.weight_xyz = weight_xyz
        self.weight_angle = weight_angle
        assert self.weight_angle == 0
        self._goal_dim = 3

    @classmethod
    def load(cls, params: OrderedDict):
        xx = params['target']
        assert len(xx) == 3
        return cls(params['name'], Pose(xx[:3]), params['weight_xyz'], params['weight_angle'])

    def _get_obs(self, sim: Simulator):
        return sim.objects[self.agent].pose.p, self.target_pose_p

    def compute_cost(self, achieved, target, info=None):
        return self.weight_xyz * np.linalg.norm(target - achieved, axis=-1).clip(0, 1)


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
    CTRLNORM = ControlNorm,
    MOVEARM = ArmMove
)


class WaypointList(Waypoint):
    def __init__(self, *args):
        object.__init__(self)
        self.args: List[Waypoint] = args

    @classmethod
    def load(cls, params: List):
        # TODO: seems very stupid
        return WaypointList(*[WAYPOINTS[waypoint[0]].load(waypoint[1]) for waypoint in params])

    def _get_obs(self, sim: Simulator):
        achieved_goal = []
        desired_goal = []
        for cost in self.args:
            achieved, target = cost._get_obs(sim)
            if achieved is not None:
                achieved_goal.append(achieved)
            if target is not None:
                desired_goal.append(target)
        return np.concatenate(achieved_goal), np.concatenate(desired_goal)

    def compute_cost(self, achieved, desired, info):
        r = 0
        for cost in self.args:
            p = [cost._goal_dim,]
            _achieved, achieved = np.split(achieved, p, axis=-1)
            assert _achieved.shape[-1] == cost._goal_dim
            _desired, desired = np.split(desired, p, axis=-1)
            r += cost.compute_cost(_achieved, _desired, info)
        return r


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
        return Trajectory(*[(WaypointList.load(i['list']), i['duration']) for i in params])


def load_waypoints(params):
    return Trajectory.load(params)
