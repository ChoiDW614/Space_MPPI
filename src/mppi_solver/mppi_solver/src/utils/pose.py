import torch
from mppi_solver.src.utils.rotation_conversions import quaternion_to_matrix, matrix_to_quaternion, matrix_to_euler_angles, quaternion_multiply, quaternion_invert

class Pose():
    def __init__(self, tensor_args=None):
        if tensor_args is not None:
            self.__pose = torch.zeros(3, **tensor_args)
            self.__orientation = torch.tensor([0.0, 0.0, 0.0, 1.0], **tensor_args)
        else:
            self.__pose = torch.zeros(3)
            self.__orientation = torch.tensor([0.0, 0.0, 0.0, 1.0])
        self.__tf = quaternion_to_matrix(self.__orientation)

    @property
    def pose(self):
        return self.__pose
    
    @pose.setter
    def pose(self, pos):
        if isinstance(pos, torch.Tensor):
            self.__pose = pos
        else:
            self.__pose = torch.tensor([pos.x, pos.y, pos.z])

    @property
    def orientation(self):
        return self.__orientation
    
    @orientation.setter
    def orientation(self, ori):
        if isinstance(ori, torch.Tensor):
            self.__orientation = ori
            self.__tf = quaternion_to_matrix(self.__orientation)
        else:
            self.__orientation = torch.tensor([ori.x, ori.y, ori.z, ori.w])
            self.__tf = quaternion_to_matrix(self.__orientation)

    @property
    def tf_return(self):
        return self.__tf

    @property
    def np_pose(self):
        return self.__pose.cpu().numpy()
    
    @property
    def np_orientation(self):
        return self.__orientation.cpu().numpy()
    
    @property
    def rpy(self):
        return matrix_to_euler_angles(self.__tf, "ZYX")

    @property
    def np_rpy(self):
        return matrix_to_euler_angles(self.__tf, "ZYX").numpy()

    @property
    def x(self):
        return self.__pose[0].item()
    
    @x.setter
    def x(self, value):
        self.__pose[0] = value

    @property
    def y(self):
        return self.__pose[1].item()
    
    @y.setter
    def y(self, value):
        self.__pose[1] = value

    @property
    def z(self):
        return self.__pose[2].item()
    
    @z.setter
    def z(self, value):
        self.__pose[2] = value

    def tf_matrix(self, tensor_args = None):
        if tensor_args is None:
            matrix = torch.eye(4)
        else:
            matrix = torch.eye(4, **tensor_args)
        matrix[0:3, 0:3] = quaternion_to_matrix(self.__orientation)
        matrix[0:3, 3] = self.__pose
        return matrix
    
    def from_matrix(self, matrix):
        self.__pose = matrix[0:3, 3]
        self.__orientation = matrix_to_quaternion(matrix[0:3, 0:3])
        self.__tf = matrix[0:3, 0:3]

    def from_rotataion_matrix(self, matrix):
        self.__orientation = matrix_to_quaternion(matrix[0:3, 0:3])
        self.__tf = matrix[0:3, 0:3]

    def __sub__(self, Ppose):
        sub_pose = Pose()
        sub_pose.pose = self.__pose - Ppose.pose
        sub_pose.orientation = quaternion_multiply(self.__orientation, quaternion_invert(Ppose.orientation))
        return sub_pose

    def __add__(self, Ppose):
        add_pose = Pose()
        add_pose.pose = self.__pose + Ppose.pose
        add_pose.orientation = quaternion_multiply(self.__orientation, Ppose.orientation)
        return add_pose
    
    def clone(self):
        new_pose = Pose()
        new_pose.pose = self.__pose.clone()
        new_pose.orientation = self.__orientation.clone()
        return new_pose


def pose_diff(ppos1: Pose, ppose2: Pose):
    pose_difference = torch.sum(torch.abs(ppos1.pose - ppose2.pose))
    orientation_difference = torch.sum(torch.abs(ppos1.orientation - ppose2.orientation))
    return pose_difference + orientation_difference

def pos_diff(ppos1: Pose, ppose2: Pose):
    pose_difference = torch.sum(torch.abs(ppos1.pose - ppose2.pose))
    return pose_difference
