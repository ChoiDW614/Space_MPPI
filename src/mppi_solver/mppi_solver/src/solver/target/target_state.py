import numpy as np
import torch
from collections import deque
from rclpy.logging import get_logger
from builtin_interfaces.msg import Time as MSG_Time

from mppi_solver.src.solver.filter.kalman_filter import SatellitePoseKalmanFilter
from mppi_solver.src.utils.rotation_conversions import euler_angles_to_matrix, matrix_to_euler_angles

from mppi_solver.src.utils.pose import Pose
from mppi_solver.src.utils.time import Time

import os
from ament_index_python.packages import get_package_share_directory
from mppi_solver.src.robot.urdfFks.urdfFk import URDFForwardKinematics

from scipy.io import savemat


class DockingInterface(object):
    def __init__(self, init_pose: Pose, predict_step: int):
        self.logger = get_logger("EKF")

        urdf_file_path = os.path.join(get_package_share_directory('mppi_solver'), "models", "ets_vii", "ets_vii.urdf")
        self.fk_ets_vii = URDFForwardKinematics(urdf_file_path, root_link='base_link', end_links = 'docking_interface')
        self.fk_ets_vii.set_mount_transformation(torch.eye(4))
        self.fk_ets_vii.set_samples_and_timesteps(1, 1, 0)

        self.docking_pose = Pose()
        self.predict_docking_pose = None

        self.base_pose = Pose()
        self.base_pose_prev = Pose()
        self.base_vel_pos = torch.tensor([0.0, 0.0, 0.0])
        self.base_vel_rpy = torch.tensor([0.0, 0.0, 0.0])

        self.esti_base_pose = Pose()
        self.esti_base_pose_prev = Pose()

        self.interface_time = Time()
        self.interface_time_prev = Time()

        self.ekf = SatellitePoseKalmanFilter(dim_x=6, dim_z=6)
        self.ekf.set_init_pose(init_pose)
        self.predict_base_pose = None
        self.predict_interface_cov = None

        self.n_predict = predict_step
        self.dt = 0.0

        self.log_docking_pose_x = np.array([])
        self.log_docking_pose_y = np.array([])
        self.log_docking_pose_z = np.array([])
        self.log_docking_predict_pose_x = np.array([])
        self.log_docking_predict_pose_y = np.array([])
        self.log_docking_predict_pose_z = np.array([])

        # self.x_history = deque(maxlen=self.n_predict)


    def update_velocity(self):
        self.dt = self.interface_time.time - self.interface_time_prev.time
        self.base_vel_pos = (self.base_pose.pose - self.base_pose_prev.pose) / self.dt
        self.base_vel_rpy = self.compute_angular_rate(self.base_pose.rpy, self.base_pose_prev.rpy)
        return
    

    def compute_angular_rate(self, euler1, euler2):
        diff = euler1 - euler2
        diff = (diff + torch.pi) % (2 * torch.pi) - torch.pi
        angular_rate = diff / self.dt
        return angular_rate


    def ekf_update(self):
        self.dt = self.interface_time.time - self.interface_time_prev.time
        self.ekf.predict_and_update(self.base_pose, self.base_vel_pos, self.base_vel_rpy, self.dt)
        self.predict_base_pose, self.predict_interface_cov = self.ekf.predict_multi_step(self.n_predict)

        self.docking_interface_pose(self.base_pose)
        self.predict_docking_interface_pose(self.predict_base_pose)

        # self.log_docking_pose_x = np.append(self.log_docking_pose_x, self.docking_pose.pose[0].item())
        # self.log_docking_pose_y = np.append(self.log_docking_pose_y, self.docking_pose.pose[1].item())
        # self.log_docking_pose_z = np.append(self.log_docking_pose_z, self.docking_pose.pose[2].item())
        # self.log_docking_predict_pose_x = np.append(self.log_docking_predict_pose_x, self.predict_docking_pose[-1,0].item())
        # self.log_docking_predict_pose_y = np.append(self.log_docking_predict_pose_y, self.predict_docking_pose[-1,1].item())
        # self.log_docking_predict_pose_z = np.append(self.log_docking_predict_pose_z, self.predict_docking_pose[-1,2].item())

        # savemat('src/mppi_solver/mppi_solver/runs/target_pose.mat', {'x': self.log_docking_pose_x, 'y': self.log_docking_pose_y, 'z': self.log_docking_pose_z})
        # savemat('src/mppi_solver/mppi_solver/runs/predict_target_pose.mat', {'predict_x': self.log_docking_predict_pose_x, 'predict_y': self.log_docking_predict_pose_y, 'predict_z': self.log_docking_predict_pose_z})

        # Check predict and actual state
        # self.x_history.append(np.hstack([self.base_pose.np_pose, self.base_pose.np_rpy]))
        # if len(self.x_history) == self.x_history.maxlen:
        #     predicted_state = self.predict_base_pose[-1, :]
        #     actual_state = self.x_history.pop()
        #     error = actual_state - predicted_state
        #     self.logger.info('act ' + str(actual_state))
        #     self.logger.info('prd ' + str(predicted_state))
        #     self.logger.info('err ' + str(error))


    def docking_interface_pose(self, base_pose: Pose):
        tf = base_pose.tf_matrix()
        T = torch.eye(4)
        T[2, 3] = -1.0

        self.docking_pose.from_matrix(tf @ T)
        return
    

    def predict_docking_interface_pose(self, base_pose: np.ndarray):
        n_step, n_x = base_pose.shape
        T = torch.eye(4).unsqueeze(0).expand(n_step, 4, 4).clone()

        T[:, 0, 3] = torch.tensor(base_pose[:, 0], dtype=torch.float32)
        T[:, 1, 3] = torch.tensor(base_pose[:, 1], dtype=torch.float32)
        T[:, 2, 3] = torch.tensor(base_pose[:, 2], dtype=torch.float32)

        roll  = torch.tensor(base_pose[:, 3], dtype=torch.float32)
        pitch = torch.tensor(base_pose[:, 4], dtype=torch.float32)
        yaw   = torch.tensor(base_pose[:, 5], dtype=torch.float32)

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

        T_translate = torch.eye(4).unsqueeze(0).expand(n_step, 4, 4).clone()
        T_translate[:, 2, 3] = -1.0
        T = torch.matmul(T, T_translate)

        # Requires verification
        self.predict_docking_pose = torch.cat([T[:, :3, 3], matrix_to_euler_angles(T[:,0:3,0:3], "ZYX")], dim=1)
        return


    @property
    def pose(self):
        return self.base_pose
    
    @pose.setter
    def pose(self, pose: Pose):
        self.base_pose = pose.copy()

    @property
    def pose_prev(self):
        return self.base_pose_prev
    
    @pose_prev.setter
    def pose_prev(self, pose: Pose):
        self.base_pose_prev = pose.copy()

    @property
    def vel_pos(self):
        return self.base_vel_pos
    
    @vel_pos.setter
    def vel_pos(self, vel: torch.Tensor):
        self.base_vel_pos = vel

    @property
    def vel_rpy(self):
        return self.base_vel_rpy
    
    @vel_rpy.setter
    def vel_rpy(self, vel: torch.Tensor):
        self.base_vel_rpy = vel

    @property
    def predict_pose(self):
        return self.predict_base_pose

    @property
    def time(self):
        return self.interface_time.time
    
    @time.setter
    def time(self, time: Time):
        if isinstance(time, Time):
            self.interface_time = time
        elif isinstance(time, MSG_Time):
            self.interface_time.time = time
        else:
            self.interface_time.time = time

    @property
    def time_prev(self):
        return self.interface_time_prev.time
    
    @time_prev.setter
    def time_prev(self, time):
        if isinstance(time, Time):
            self.interface_time_prev = time
        elif isinstance(time, MSG_Time):
            self.interface_time_prev.time = time
        else:
            self.interface_time_prev.time = time


