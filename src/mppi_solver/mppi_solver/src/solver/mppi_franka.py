# Python 
import os
import math
import yaml
import time
from datetime import datetime

# Linear Algebra
import numpy as np
import torch

# RCLPY 
from rclpy.logging import get_logger
from ament_index_python.packages import get_package_share_directory

# Sampling Library
from mppi_solver.src.solver.sampling.gaussian_noise import GaussianSample
from mppi_solver.src.solver.sampling.standard_normal_noise import StandardSamplling


# Cost Library
from mppi_solver.src.solver.cost.cost_manager import CostManager


# FK Library
from mppi_solver.src.robot.urdfFks.urdfFk import URDFForwardKinematics

# TF Library
from mppi_solver.src.utils.pose import Pose, pose_diff, pos_diff
from mppi_solver.src.utils.rotation_conversions import euler_angles_to_matrix, matrix_to_euler_angles, quaternion_to_matrix, matrix_to_quaternion

# Filter : MPPI
from mppi_solver.src.solver.filter.svg_filter import SavGolFilter


class MPPI():
    def __init__(self):
        self.logger = get_logger("MPPI")

        # torch env
        os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info('Device: ' + self.device.type)
        torch.set_default_dtype(torch.float32)


        self.n_action = 7
        self.n_manipulator_dof = 7
        self.n_mobile_dof = 0
        self.n_samples = 1000
        self.n_horizen = 32
        self.dt = 0.01

        # Manipulator states
        self._q = torch.zeros(self.n_action, device=self.device)
        self._qdot = torch.zeros(self.n_action, device=self.device)
        self._qddot = torch.zeros(self.n_action, device=self.device)

        self.q_prev = torch.zeros(self.n_action, device=self.device)
        self.v_prevv = torch.zeros(self.n_action, device=self.device)


        self.ee_pose = Pose()
        self.eefTraj = torch.zeros((self.n_samples, self.n_horizen, 4, 4), device=self.device)

        # Action
        self.u = torch.zeros((self.n_action), device=self.device)
        self.u_prev = torch.zeros((self.n_horizen, self.n_action), device = self.device)

        # Buffer
        self.buffer_size = 10
        self.weight_buffer = torch.zeros((self.buffer_size, self.n_samples), device=self.device)
        self.action_buffer = torch.zeros((self.buffer_size, self.n_samples, self.n_horizen, self.n_action), device=self.device)

        # Sampling class
        # self.sample_gen = GaussianSample(self.n_horizen, self.n_action, self.buffer_size, device= self.device)
        self.sample_gen = StandardSamplling(self.n_samples, self.n_horizen, self.n_action, self.device)

        # base control states
        self.base_pose = Pose()

        # Target states
        self.target_pose = Pose()
        self.target_pose.pose = torch.tensor([0.5, 0.0, 0.6])
        self.target_pose.orientation = torch.tensor([0.0, 0.0, 0.0, 1.0])
        # self.target_pose.orientation = torch.tensor([0, -0.4871745, 0, -0.8733046])
        self.predict_target_pose = torch.zeros((self.n_horizen, 6))
        
        self._lambda = 0.1
        self.cost_manager = CostManager(self.n_samples, self.n_horizen, self.n_action, self._lambda, self.device)

        # Import URDF for forward kinematics
        package_name = "mppi_solver"
        urdf_file_path = os.path.join(get_package_share_directory(package_name), "models", "franka", "franka.urdf")

        self.fk_canadarm = URDFForwardKinematics(urdf_file_path, root_link='panda_link0', end_links = 'panda_link7')

        self.isBaseMoving = False
        mount_tf = torch.eye(4, device=self.device)
        self.fk_canadarm.set_mount_transformation(mount_tf)
        self.fk_canadarm.set_samples_and_timesteps(self.n_samples, self.n_horizen, self.n_mobile_dof)

        self.svg_filter = SavGolFilter()

        # Log
        self.cnt = 0
        
    def check_reach(self):
        self.ee_pose.from_matrix(self.fk_canadarm.forward_kinematics_cpu(self._q[self.n_mobile_dof:], 'panda_link7', 'panda_link0', init_transformation=None, base_movement=False))
        pose_err = pos_diff(self.ee_pose, self.target_pose)
        ee_ori_mat = euler_angles_to_matrix(self.ee_pose.rpy, "ZYX")
        target_ori_mat = euler_angles_to_matrix(self.target_pose.rpy, "ZYX")
        diff_ori_mat = torch.matmul(torch.linalg.inv(ee_ori_mat), target_ori_mat)
        diff_ori_quat = matrix_to_quaternion(diff_ori_mat)
        diff_ori_3d = matrix_to_euler_angles(diff_ori_mat, "ZYX")

        self.logger.info(f"Pose Err : {pose_err}")
        self.logger.info(f"Ori Err : {diff_ori_3d}")
        if pose_err < 0.005:
            return True
        else:
            return False

    def compute_control_input(self):
        if self.check_reach():
            return self.qdes, self.vdes
        u = self.u_prev.clone()
        noise = self.sample_gen.sampling()
        v = u + noise
        qSamples = self.sample_gen.get_sample_joint(v, self._q, self._qdot, self.dt)
        trajectory = self.fk_canadarm.forward_kinematics(qSamples, 'panda_link7', 'panda_link0', self.base_pose.tf_matrix(self.device), base_movement=self.isBaseMoving)

        none_joint_trajs = torch.zeros((self.n_samples, self.n_horizen, self.n_action), device=self.device)
        self.cost_manager.update_pose_cost(qSamples, v, trajectory, none_joint_trajs, self.target_pose)
        self.cost_manager.update_covar_cost(u, v, self.sample_gen.sigma_matrix)
        S = self.cost_manager.compute_all_cost()

        w = self.compute_weights(S, self._lambda)
        w_expanded = w.view(-1, 1,1)
        w_eps = torch.sum(w_expanded * noise, dim = 0)
        w_eps = self.svg_filter.savgol_filter_torch(w_eps,window_size=9,polyorder=2)

        u+= w_eps

        self.u_prev = u.clone()
        self.u = u[0].clone()

        self.vdes = self._qdot + self.u * self.dt
        self.qdes = self._q + self._qddot * self.dt + 0.5 * self.u * self.dt * self.dt

        return self.qdes, self.vdes




    def compute_weights(self, S: torch.Tensor, _lambda) -> torch.Tensor:
        """
        Compute weights for each sample in a batch using PyTorch.
        
        Args:
            S (torch.Tensor): Tensor of shape (batch_size,) containing the scores (costs) for each sample.

        Returns:
            torch.Tensor: Tensor of shape (batch_size,) containing the computed weights.
        """
        # 최소값 계산 (rho)
        rho = S.min()  # (scalar)

        # eta 계산
        scaled_S = (-1.0 / _lambda) * (S - rho)  # (batch_size,)
        eta = torch.exp(scaled_S).sum()  # (scalar)

        # 각 샘플의 weight 계산
        weights = torch.exp(scaled_S) / eta  # (batch_size,)

        return weights

    # def getSampleJoint(self, samples):
    #     # samples: (n_sample, n_horizon, n_action)
    #     n_sample, n_horizon, n_action = samples.shape

    #     # 초기 속도와 위치 확장
    #     qdot0 = self._qdot.unsqueeze(0).unsqueeze(0).expand(n_sample, 1, n_action)  # (n_sample, 1, n_action)
    #     q0 = self._q.unsqueeze(0).unsqueeze(0).expand(n_sample, 1, n_action)        # (n_sample, 1, n_action)
    #     v = torch.cumsum(samples * self.dt, dim=1) + qdot0  # (n_sample, n_horizon, n_action)

    #     # 이전 속도: [v0, v0+..., ..., v_{N-1}]
    #     v_prev = torch.cat([qdot0, v[:, :-1, :]], dim=1)  # (n_sample, n_horizon, n_action)

    #     # 누적 위치 계산: q[i] = q[i-1] + v[i-1] * dt + 0.5 * a[i] * dt^2
    #     dq = v_prev * self.dt + 0.5 * samples * self.dt**2
    #     q = torch.cumsum(dq, dim=1) + q0

    #     return q

    def set_joint(self, joint_states):
        joint_states = joint_states.to(self.device)
    
        self._q = joint_states[:, 0]
        self._qdot = joint_states[:, 1]
        self._qddot = joint_states[:, 2]
        return