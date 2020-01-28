import numpy as np

#from gym.envs.robotics import rotations, robot_env, utils
from . import robot_env
from .robot_env import sapien_core, Pose
from . import robot_utils
import warnings


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class MovoEnv(robot_env.RobotEnv):
    def __init__(
            self, model_path, n_substeps, gripper_extra_height=None, block_gripper=True,
            has_object=False, target_in_the_air=False, target_offset=0.0, obj_range=0.15, target_range=0.15,
            distance_threshold=0.01, initial_qpos=None, reward_type=None,
    ):
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.target_range = target_range
        self.obj_range = obj_range
        self.target_offset = target_offset
        self.target_in_the_air = target_in_the_air
        # n_atcions = 4? because it's position control
        super(MovoEnv, self).__init__(model_path, initial_qpos=initial_qpos, n_substeps=n_substeps, n_actions=4)

    def _env_setup(self, initial_qpos):
        # What we should do:
        #    1. set the joint qpos
        #    2. move effector into position
        #    3. changing the location of the
        self.model.set_qpos(initial_qpos)

        for _ in range(10):
            self.sim.step()

        for i in self.model.get_links():
            if i.name == 'right_gripper_base_link':
                self.gripper_link = i
        self.initial_gripper_xpos = np.concatenate((self.gripper_link.pose.p, self.gripper_link.pose.q))

        l, r, count = 1000, 0, 0
        for i in self.model.get_joints():
            if 'gripper' in i.name:
                l = min(l, count)
                r = max(r, count + i.get_dof())
            count += i.get_dof()
        self.gripper_idx = slice(l, r)

        if self.has_object:
            self.height_offset = self.obj.pose.p[2]

    def _reset_sim(self):
        # set robot state
        self.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)

            object_qpos = self.scene.get_qpos()

            #TODO: in the current implementation, the box has no orientation...
            assert object_qpos.shape == (3,) #object qpos is 3
            object_qpos[:2] = object_xpos
            self.scene.set_qpos(object_qpos)

        self.sim.step()
        return True

    def _sample_goal(self):
        #goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _load_robot(self, urdf_path: str, material: sapien_core.PxMaterial) -> None:
        # By default, the robot will loaded with balanced passive force
        #self.loader.fix_base = True
        self.loader.fix_root_link = True
        model: sapien_core.Articulation = self.loader.load(urdf_path, material)
        model.set_root_pose(sapien_core.Pose([0, 0, 0], [1, 0, 0, 0]))

        # Link mapping, remember to set the self._base_link_name if your robot base is not that name
        joints = model.get_joints()
        self._q_names = [j.name for j in joints if j.get_dof() > 0]
        self._base_link_name = "base_link"
        self._dof = sum([j.get_dof() for j in joints])

        # Setup actuator
        #for qname, joint in zip(self._q_names, joints):

        for joint in joints:
            qname = joint.name
            flag = 'right' in qname
            if self.block_gripper and 'gripper' in qname:
                flag = False
            if flag:
                self.add_force_actuator(qname, -50, 50)
            elif joint.get_dof() > 0:
                joint.set_drive_property(1000000, 1000000)
                joint.set_drive_target(0 if 'gripper' not in qname else 0.986)
                #joint.set_limits(np.array([[0, 0.0001]]))
        return model

    def _load_scene(self) -> None:
        PxIdentity = np.array([1, 0, 0, 0])
        x2y = np.array([0.7071068, 0, 0, 0.7071068])
        x2z = np.array([0.7071068, 0, 0.7071068, 0])

        tabel_heigh = 0.4

        x = 0.9
        table = self.add_link(None,  Pose(np.array([0, 0, 0]), PxIdentity), "world", contype=1, conaffinity=1) # root coordinates #free
        table.add_box_shape(Pose([x, 0, tabel_heigh/2]), np.array((0.5, 0.8, tabel_heigh/2)))
        table.add_box_visual(Pose([x, 0, tabel_heigh/2]), size=np.array((0.5, 0.8, tabel_heigh/2)), color=np.array([0., 0., 1.]))

        if self.has_object:
            size = 0.05
            size = 0.1
            obj_slidey = self.my_add_link(table, ([x, 0., tabel_heigh + size], PxIdentity), ((0, 0, 0), x2y), "obj_slidey", "obj_slidery", [-2, 2], damping=0.1, type='slider')
            obj_slidez = self.my_add_link(obj_slidey, ([0, 0, 0], PxIdentity), ((0, 0, 0), PxIdentity), "obj_slidex", "obj_sliderx", [-2, 2], damping=0.1, type='slider')
            obj = self.my_add_link(obj_slidez, ([0.0, 0.0, 0.0], PxIdentity), ((0, 0, 0), x2z), "obj", "obj_sliderz", [-2, 2], damping=0.1, type='slider', contype=1)
            self.add_box(obj, (0, 0, 0), PxIdentity, (size, size, size), (1, 0, 0), "object", density=0.000001)

        scene = self.builder.build(True)
        scene.set_root_pose(Pose([0., 0., 0.]))
        self.obj = scene.get_links()[-1]
        return scene

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def _step_callback(self):
        # 13 dof
        # 180 6x link num
        #linear_velocity angular_velocy
        #print(self.model.compute_jacobian().shape)
        #exit(0)
        pass

    def _set_action(self, a):
        assert len(a) == len(self._actuator_index), "Action dimension must equal to the number of actuator"
        qf = np.zeros(self._dof)
        qf[self._actuator_index] = a
        self.model.set_qf(qf)

    def _get_obs(self):

        grip_pos = np.concatenate((self.gripper_link.pose.p, self.gripper_link.pose.q))[:3]
        dt = self.dt

        grip_velp = np.concatenate((self.gripper_link.velocity, self.gripper_link.angular_velocity))[:3] * dt

        robot_qpos, robot_qvel = self.model.get_qpos(), self.model.get_qvel()
        if self.has_object:
            object_pos, object_rot = self.obj.pose.p, self.obj.pose.q
            object_velp = self.obj.velocity * dt
            object_velr = self.obj.angular_velocity * dt

            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[self.gripper_idx]
        gripper_vel = robot_qvel[self.gripper_idx] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
            #raise NotImplementedError

        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        self.sim.set_ambient_light([.4, .4, .4])
        self.sim.set_shadow_light([1, -1, -1], [.5, .5, .5])
        self.sim.add_point_light([2, 2, 2], [1, 1, 1])
        self.sim.add_point_light([2, -2, 2], [1, 1, 1])
        self.sim.add_point_light([-2, 0, 2], [1, 1, 1])

        # MODE 1
        self.viewer.set_camera_position(3, -1.5, 1.65)
        self.viewer.set_camera_rotation(-3.14-0.5, -0.2)

        # MODE 2
        #self.viewer.camera.set_position([1.3, -3, 1.65])
        #self.viewer.camera.rotate_yaw_pitch(-3.14 - 1.57, -0.2)

        self.viewer.set_current_scene(self.sim)
