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
        cost = 0
        if self.weight_contact > 0:
            cost = self.arm_contact_cost(sim, self.contact_epsilon) * self.weight_contact
        return  cost + xyz * self.weight_xyz + theta * self.weight_angle


class Grasped(Waypoint):
    def __init__(self, agent, obj_name, epsilon=0.01, range=0.1):
        super(Grasped, self).__init__(agent)

        self.obj_name = obj_name
        self.epsilon = epsilon

    def cost(self, sim: Simulator):
        agent = sim.objects[self.agent]

        def is_finger(actor):
            if actor.name == 'right_gripper_figner1_finger_tip_link':
                return 1
            if actor.name == 'right_gripper_figner2_finger_tip_link':
                return 2
            if actor.name == 'right_gripper_figner3_finger_tip_link':
                return 3
            return 0

        """
        def is_object(actor):
            #if not isinstance(actor, sap)
            if actor.get_articulation().name != self.obj_name:
                return False
            return True
        
        for contact in sim.scene.get_contacts():
            finger_id_1 = is_finger(contact.actor_1)
            is_object_1 = is_object(contact.actor_1)

            finger_id_2 = is_finger(contact.actor_2)
            is_object_2 = is_object(contact.actor_2)

            if finger_id_1 and is_object_2:
                mm[finger_id_1-1] = min(mm[finger_id_1-1], contact.separation)
            if finger_id_2 and is_object_1:
                mm[finger_id_2-1] = min(mm[finger_id_2-1], contact.separation)
        """

        mm = np.zeros(3) + np.inf
        object = sim.objects[self.obj_name]
        cmass = object.pose * object.cmass_local_pose
        for link in agent.get_links():
            finger_id = is_finger(link)
            if finger_id:
                link_cmass = link.pose * link.cmass_local_pose
                mm[finger_id-1] = min(mm[finger_id-1], np.linalg.norm(cmass.p - link_cmass.p))
        return -mm.sum() # 中心的位置。。


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
        return  xyz * self.weight_xyz + theta * self.weight_angle



class WaypointList(Waypoint):
    def __init__(self, *args):
        object.__init__(self)
        self.args: List[Waypoint] = args

    def cost(self, sim: Simulator):
        total = 0
        for i in self.args:
            total += i.cost(sim)
        return total
