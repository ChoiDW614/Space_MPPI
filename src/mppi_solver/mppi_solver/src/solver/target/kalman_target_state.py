import torch
import numpy as np

import os
from rclpy.logging import get_logger

from collections import deque
from builtin_interfaces.msg import Time as MSG_Time
from geometry_msgs.msg import TransformStamped, PoseStamped

from mppi_solver.src.solver.filter.kalman_filter import PoseKalmanFilter
from mppi_solver.src.utils.rotation_conversions import euler_angles_to_matrix, matrix_to_euler_angles

from mppi_solver.src.utils.pose import Pose
from mppi_solver.src.utils.time import Time

from ament_index_python.packages import get_package_share_directory

from pathlib import Path
from mppi_solver.src.utils.matlab_logger import MATLABLogger


class DockingInterface(object):
    def __init__(self, controller_ref=None, predict_step: int=32):
        self.logger = get_logger("docking_interface")

        self.sim_time = Time()
        self.sim_time_prev = Time()

        self.align_pose = Pose()
        self.esti_docking_pose = Pose()
        self.esti_docking_pose_prev = Pose()

        # kalman filter
        self.ekf = PoseKalmanFilter(dim_x=6, dim_z=6)

        self.vel_pos = torch.tensor([0.0, 0.0, 0.0])
        self.vel_rpy = torch.tensor([0.0, 0.0, 0.0])
        self.vel_pos_prev = None
        self.vel_rpy_prev = None
        self.predict_docking_pose = None
        self.predict_interface_cov = None
        self.predict_pose_list = list()
        self.predict_conv_list = list()

        self.n_predict = predict_step
        self.dt = 0.0
        self.is_initialize_kalman = False

        # True docking pose
        self.true_docking_pose= Pose()
        self.true_docking_pose_prev = Pose()
        self.tmp_pose = Pose()
        self.docking_transformation_matrix = torch.eye(4)
        self.docking_transformation_matrix[2, 3] = -1.0
        self.true_docking_pose_kalman = Pose()

        # test
        self.vel_deque = deque(maxlen=32)
        self.vel_deque.append(np.array([0,0,0,0,0,0], dtype=np.float32))

        # MPPI Controller reference
        self._controller_ref = controller_ref

        # Log
        self.matlab_logger = MATLABLogger(script_name=Path(__file__).stem, file_name="target_pose")
        self.matlab_logger.create_dataset(dataset_name="true_pose", shape=7)
        self.matlab_logger.create_dataset(dataset_name="kalman_pose", shape=7)
        self.matlab_logger.create_dataset(dataset_name="vel", shape=7)
        self.matlab_logger.create_dataset(dataset_name="vel_mean", shape=7)


    def update_velocity(self):
        self.dt = self.sim_time.time - self.sim_time_prev.time

        self.compute_velocity(self.true_docking_pose, self.true_docking_pose_prev)
        self.matlab_logger.log("vel", [self.sim_time.time] + self.vel_pos.tolist() + self.vel_rpy.tolist())

        # test
        self.vel_deque.append(np.concatenate([self.vel_pos.numpy(), self.vel_rpy.numpy()]))
        self.vel_mean = np.mean(np.stack(self.vel_deque, axis=0), axis=0)
        self.matlab_logger.log("vel_mean", [self.sim_time.time] + self.vel_mean.tolist())

        # prev state
        self.true_docking_pose_prev = self.true_docking_pose.clone()
        self.sim_time_prev.time = self.sim_time.time
        return
    

    def compute_velocity(self, pose: Pose, pose_prev: Pose):
        if self.vel_pos_prev is None:
            self.vel_pos_prev = torch.zeros_like(pose.pose)

        self.vel_pos = (pose.pose - pose_prev.pose) / self.dt
        self.vel_pos_prev = self.vel_pos.clone()

        if self.vel_rpy_prev is None:
            self.vel_rpy_prev = pose.rpy.clone()
        
        diff = pose.rpy - self.vel_rpy_prev
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))

        self.vel_rpy_prev += diff
        self.vel_rpy = diff / self.dt
        return


    def ekf_update(self):
        # calculate velocity
        if self.is_initialize_kalman:
            self.update_velocity()
            self.true_docking_pose_kalman = self.ekf.predict_and_update(self.true_docking_pose, self.vel_pos, self.vel_rpy, self.dt)
            self.predict_docking_pose, self.predict_interface_cov = self.ekf.predict_multi_step(self.n_predict)
            self.array_to_predict_pose_list()
            self.matlab_logger.log("true_pose", [self.sim_time.time] + self.true_docking_pose.np_pose.tolist() + self.true_docking_pose.np_rpy.tolist())
            self.matlab_logger.log("kalman_pose", [self.sim_time.time] + self.true_docking_pose_kalman.np_pose.tolist() + self.true_docking_pose_kalman.np_rpy.tolist())


    def set_true_docking_pose(self, pose: TransformStamped):
        self.tmp_pose = Pose()
        self.tmp_pose.pose = pose.transform.translation
        self.tmp_pose.orientation = pose.transform.rotation

        if not self.is_initialize_kalman:
            self.initialize_kalman_filter()

        self.true_docking_pose.from_matrix(self.tmp_pose.tf_matrix() @ self.docking_transformation_matrix)
        return
    

    def set_esti_docking_pose(self, pose: PoseStamped):
        self.time = pose.header.stamp
        self.esti_docking_pose.pose = pose.pose.position
        self.esti_docking_pose.orientation = pose.pose.orientation
        return


    def initialize_kalman_filter(self):
        if torch.allclose(self.true_docking_pose.pose, torch.tensor([0.0, 0.0, 0.0])) \
            and torch.allclose(self.true_docking_pose.orientation, torch.tensor([0.0, 0.0, 0.0, 1.0])):
            return
        else:
            self.ekf.set_init_pose(self.true_docking_pose)
            self.is_initialize_kalman = True
        return
    

    def array_to_predict_pose_list(self):
        n_step, n_x = self.predict_docking_pose.shape
        T = torch.eye(4).unsqueeze(0).expand(n_step, 4, 4).clone()

        T[:, 0, 3] = torch.tensor(self.predict_docking_pose[:, 0], dtype=torch.float32)
        T[:, 1, 3] = torch.tensor(self.predict_docking_pose[:, 1], dtype=torch.float32)
        T[:, 2, 3] = torch.tensor(self.predict_docking_pose[:, 2], dtype=torch.float32)

        roll  = torch.tensor(self.predict_docking_pose[:, 3], dtype=torch.float32)
        pitch = torch.tensor(self.predict_docking_pose[:, 4], dtype=torch.float32)
        yaw   = torch.tensor(self.predict_docking_pose[:, 5], dtype=torch.float32)

        cr = torch.cos(roll)
        sr = torch.sin(roll)
        cp = torch.cos(pitch)
        sp = torch.sin(pitch)
        cy = torch.cos(yaw)
        sy = torch.sin(yaw)

        R00 = cy * cp
        R01 = cy * sp * sr - sy * cr
        R02 = cy * sp * cr + sy * sr

        R10 = sy * cp
        R11 = sy * sp * sr + cy * cr
        R12 = sy * sp * cr - cy * sr

        R20 = -sp
        R21 = cp * sr
        R22 = cp * cr

        R = torch.stack([
            torch.stack([R00, R01, R02], dim=1),
            torch.stack([R10, R11, R12], dim=1),
            torch.stack([R20, R21, R22], dim=1)
            ], dim=1)
        T[:, :3, :3] = R
        # pose_torch = torch.cat([T[:, :3, 3], matrix_to_euler_angles(T[:,0:3,0:3], "ZYX")], dim=1)

        self.predict_pose_list = []
        self.predict_conv_list = []
        for i in range(n_step):
            p = Pose()
            p.pose = T[i,:3,3]
            p.from_rotataion_matrix(T[i,0:3,0:3])
            self.predict_pose_list.append(p)
            self.predict_conv_list.append(torch.from_numpy(self.predict_interface_cov[i,:,:]))
        return
    

    @property
    def align_docking_pose(self):
        T = torch.eye(4)
        # T[2, 3] = -1.5
        self.align_pose.from_matrix(self.true_docking_pose.tf_matrix() @ T)
        return self.align_pose
    
    @property
    def true_align_docking_pose(self):
        if not self._controller_ref.is_align:
            T = torch.eye(4)
            self.align_pose.from_matrix(self.true_docking_pose.tf_matrix() @ T)
            # self.logger.info("not align docking pose")
        else:
            self.align_pose.from_matrix(self.true_docking_pose.tf_matrix())
            # self.logger.info(f"sim_time: {self.sim_time.time}, align docking pose")
            # self.logger.info("align docking pose")
        return self.align_pose

    @property
    def pose(self):
        return self.true_docking_pose.clone()
    
    @pose.setter
    def pose(self, pose: Pose):
        self.true_docking_pose = pose.clone()

    @property
    def pose_prev(self):
        return self.true_docking_pose_prev.clone()
    
    @pose_prev.setter
    def pose_prev(self, pose: Pose):
        self.true_docking_pose_prev = pose.clone()

    @property
    def predict_pose(self):
        return self.predict_pose_list
    
    @property
    def predict_conv(self):
        return self.predict_conv_list

    @property
    def time(self):
        return self.sim_time.time
    
    @time.setter
    def time(self, time: Time):
        if isinstance(time, Time):
            self.sim_time = time
        elif isinstance(time, MSG_Time):
            self.sim_time.time = time
        else:
            self.sim_time.time = time

    @property
    def time_prev(self):
        return self.sim_time.time
    
    @time_prev.setter
    def time_prev(self, time):
        if isinstance(time, Time):
            self.sim_time = time
        elif isinstance(time, MSG_Time):
            self.sim_time.time = time
        else:
            self.sim_time.time = time

