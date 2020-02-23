# here we define the action set of the robot arm
from typing import List
import numpy as np
from transforms3d import quaternions
from .simulator import Simulator
from sapien.core import Pose


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
        return self.arm_contact_cost(sim, self.contact_epsilon) * self.weight_contact +\
               xyz * self.weight_xyz + theta * self.weight_angle


class WaypointList(Waypoint):
    def __init__(self, *args):
        object.__init__(self)
        self.args: List[Waypoint] = args

    def cost(self, sim: Simulator):
        total = 0
        for i in self.args:
            total += i.cost(sim)
        return total
