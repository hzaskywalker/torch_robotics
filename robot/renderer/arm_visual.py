from .. import torch_robotics as tr

def totensor(x):
    import torch
    return torch.tensor(x, dtype=torch.float64)

class Arm:
    # currently we only consider the robot arm
    # we will consider the general articulator later

    def __init__(self, scene, M, A):
        self.scene = scene
        self.M = totensor(M)[None,:]
        self.A = totensor(A)[None,:]
        self.shapes = []
        self.shape_pose = []

    def add_shapes(self, shape, local_pose):
        # Note the pose is in the local frame
        self.shapes.append(shape)
        if local_pose is not None:
            local_pose = totensor(local_pose)[None,:]
        self.shape_pose.append(local_pose)

    def fk(self, q):
        q = totensor(q)[None,:]
        return tr.fk_in_space(q, self.M, self.A)[0] # (n+1, 4, 4)

    def set_pose(self, q):
        # Notice that set pose is different to set qpos
        Ts = self.fk(q)
        for T, shape, shape_pose in zip(Ts, self.shapes, self.shape_pose):
            if shape is None:
                continue
            pose = tr.dot(T[None,:], shape_pose).detach().numpy()[0]
            shape.set_pose(pose)
