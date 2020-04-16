from robot import tr

def link2body(link):
    import numpy as np
    cmass_pose = link.get_pose() * link.cmass_local_pose
    cmass = tr.pose2SE3(cmass_pose)
    cmass = tr.togpu(cmass)

    inertia = np.zeros((3, 3), dtype=np.float64)
    inertia[0, 0] = link.inertia[0]
    inertia[1, 1] = link.inertia[1]
    inertia[2, 2] = link.inertia[2]
    inertia = tr.togpu(inertia)
    m = tr.togpu(link.get_mass())
    return tr.RigidBody(cmass, inertia, m)

class LinkGroup:
    # group a set of robot arm as
    def __init__(self, links):
        self.links = links
        self._rigid_body = self.get_rigid_body()

        self._rigid_body.align_principle_()
        self.inertia = tr.tocpu(self._rigid_body.inertia)[0][[0, 1, 2], [0, 1, 2]]
        self.mass = tr.tocpu(self._rigid_body.mass)[0]
        self.cmass_local_pose = self.compute_cmass_local_pose()

    def get_mass(self):
        return self.mass

    def get_rigid_body(self):
        bodies = []
        for l in self.links:
            bodies.append(link2body(l))
        total = bodies[0].sum_boides(bodies)
        return total

    def get_pose(self):
        return self.links[0].get_pose()

    def compute_cmass_local_pose(self):
        cmass_space = tr.tocpu(self._rigid_body.cmass[0])
        cmass_local = tr.pose2SE3(self.get_pose().inv()) @ cmass_space
        import transforms3d
        from sapien.core import Pose
        q = transforms3d.quaternions.mat2quat(cmass_local[:3,:3])
        p = cmass_local[:3,3]
        return Pose(p, q)



class Arm7DOF:
    # a wrapper to the robot arm so that we only need to consider 7 dof
    def __init__(self, agent, dofs, ee_idx, joints_id):
        self.dofs = dofs
        self.agent = agent
        self.ee_link_idx = ee_idx
        self.joints_id = joints_id
        #self.get_links()
        links = self.agent.get_links()
        self.set_qpos(self.get_qpos()*0)
        """
        for idx, i in enumerate(self.agent.get_links()):
            print(idx, i.name)
        for i in range(13, 22):
            print(links[i].name)
        exit(0)
        """
        self.last_link = LinkGroup([links[i] for i in range(13, 22)])


    def get_qpos(self):
        return self.agent.get_qpos()[self.dofs]

    def get_qvel(self):
        return self.agent.get_qvel()[self.dofs]

    def get_qf(self):
        return self.agent.get_qf()[self.dofs]

    def get_qacc(self):
        return self.agent.get_qacc()[self.dofs]

    def set_qpos(self, qpos):
        q = self.agent.get_qpos() * 0
        q[self.dofs] = qpos
        self.agent.set_qpos(q)

    def set_qvel(self, qvel):
        q = self.agent.get_qvel() * 0
        q[self.dofs] = qvel
        self.agent.set_qvel(q)

    def set_qf(self, qf):
        q = self.agent.get_qf() * 0
        q[self.dofs] = qf
        self.agent.set_qf(q)

    def compute_jacobian(self):
        return self.agent.compute_jacobian()[self.ee_link_idx*6:self.ee_link_idx*6+6, self.dofs] # in joint space

    def compute_inverse_dynamics(self, qacc):
        q = self.agent.get_qacc() * 0
        q[self.dofs] = qacc
        return self.agent.compute_inverse_dynamics(q)[self.dofs]

    def get_ee_links(self):
        return self.agent.get_links()[self.ee_link_idx]

    def get_joints(self):
        joints = self.agent.get_joints()
        return [joints[i] for i in self.joints_id]

    def get_links(self):
        # return dof+1 links
        joints = self.agent.get_joints()
        joints = [joints[i] for i in self.joints_id]
        links = []
        for i in joints:
            links.append(i.get_parent_link())
        #links.append(joints[-1].get_child_link())
        links.append(self.last_link)
        return links

    def compute_passive_force(self, gravity=True, coriolisAndCentrifugal=True, external=True):
        return self.agent.compute_passive_force(external=external, gravity=gravity,
                                                coriolisAndCentrifugal=coriolisAndCentrifugal)[self.dofs]
