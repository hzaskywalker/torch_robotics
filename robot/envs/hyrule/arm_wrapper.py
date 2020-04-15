class Arm7DOF:
    # a wrapper to the robot arm so that we only need to consider 7 dof
    def __init__(self, agent, dofs, ee_idx, joints_id):
        self.dofs = dofs
        self.agent = agent
        self.ee_link_idx = ee_idx
        self.joints_id = joints_id

    def get_qpos(self):
        return self.agent.get_qpos()[self.dofs]

    def get_qvel(self):
        return self.agent.get_qvel()[self.dofs]

    def get_qf(self):
        return self.agent.get_qf()[self.dofs]

    def get_qacc(self):
        return self.agent.get_qacc()[self.dofs]

    def set_qpos(self, qpos):
        q = self.agent.get_qpos()
        q[self.dofs] = qpos
        self.agent.set_qpos(q)

    def set_qvel(self, qvel):
        q = self.agent.get_qvel()
        q[self.dofs] = qvel
        self.agent.set_qvel(q)

    def set_qf(self, qf):
        q = self.agent.get_qf()
        q[self.dofs] = qf
        self.agent.set_qf(q)

    def compute_jacobian(self):
        return self.agent.compute_jacobian()[self.ee_link_idx*6:self.ee_link_idx*6+6, self.dofs] # in joint space

    def compute_passive_force(self):
        return self.agent.compute_passive_force()[self.dofs]

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
        links.append(joints[-1].get_child_link())
        return links

