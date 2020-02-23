import numpy as np
from transforms3d import quaternions
from .gameplay import Constraint, Simulator

def calc_dist(A, B):
    xyz_dist = np.linalg.norm(A.p - B.p)
    theta = quaternions.quat2axangle(quaternions.qmult(A.q, quaternions.qinverse(B.q)))[1]
    return xyz_dist , theta


class Stablize(Constraint):
    def __init__(self, eps=0.01):
        super(Stablize, self).__init__(1, False)
        self.eps = eps

    def postprocess(self, sim):
        for i in range(20):
            sim.step_scene()

    def cost(self, sim_t: Simulator, s, t):
        for i in range(10):
            sim_t.step_scene()
        t_2 = sim_t.state_dict()
        sim_t.load_state_dict(t)
        out = 0
        for a in t:
            a, b = calc_dist(t[a]['pose'], t_2[a]['pose'])
            out += a + b * 0.5
        return out / self.eps



class GeoConstraint(Constraint):
    def __init__(self, obj1_name, obj2_name, pose, range, norm=None):
        super(GeoConstraint, self).__init__(5, perpetual=True)
        self.pose = pose
        self.norm = norm
        self.range = range
        self.obj1_name = obj1_name
        self.obj2_name = obj2_name

    def cost(self, sim_t, s, t):
        pose1 = t[self.obj1_name]['pose']
        pose2 = t[self.obj2_name]['pose']
        diff = calc_dist(pose2 * pose1.inv() , self.pose)
        return diff[0]/self.range[0] + diff[1]/self.range[1]
