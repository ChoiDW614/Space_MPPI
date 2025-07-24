import torch
import torch.nn as nn
import numpy as np

import time

from mppi_solver.src.utils.pose import Pose
from mppi_solver.src.utils.rotation_conversions import euler_angles_to_matrix, matrix_to_euler_angles
from mppi_solver.src.utils.rotation_conversions import matrix_to_quaternion, quaternion_invert, quaternion_multiply, quaternion_to_axis_angle

from rclpy.logging import get_logger

class PoseCost():
    def __init__(self, params, gamma, n_horizon, tensor_args):
        self.logger = get_logger("PoseCost")
        self.tensor_args = tensor_args
        self.n_horizon = n_horizon
        self.gamma = gamma
        self.gamma_horizon_gpu = self.gamma ** torch.arange(self.n_horizon-1, **self.tensor_args)
        self.gamma_horizon_cpu = self.gamma ** torch.arange(self.n_horizon)

        self.stage_pose_weight = params['stage_pose_weight']
        self.stage_orientation_weight = params['stage_orientation_weight']
        
        self.terminal_pose_weight = params['terminal_pose_weight']
        self.terminal_orientation_weight = params['terminal_orientation_weight']
        

    def compute_stage_cost(self, eefTraj: torch.Tensor, target_pose: Pose) -> torch.Tensor:
        # self.logger.info(f"eetraj{eefTraj.shape}")
        ee_sample_pose = eefTraj[:,:-1,0:3,3]
        ee_sample_orientation = eefTraj[:,:-1,0:3,0:3]

        target_pose_tensor = target_pose.pose.to(**self.tensor_args)
        diff_pose = ee_sample_pose - target_pose_tensor

        # ee_sample_quat = matrix_to_quaternion(ee_sample_orientation)
        # q_target = target_pose.orientation.to(**self.tensor_args)
        # q_target = q_target.view(1, 1, 4).expand_as(ee_sample_quat)
        # ee_quat_inv = quaternion_invert(ee_sample_quat)
        # q_diff = quaternion_multiply(q_target, ee_quat_inv) 
        # diff_orientation = quaternion_to_axis_angle(q_diff)

        target_pose_ori_mat = euler_angles_to_matrix(target_pose.rpy.to(**self.tensor_args), "ZYX")

        diff_ori_mat = torch.matmul(torch.linalg.inv(ee_sample_orientation), target_pose_ori_mat)
        diff_orientation = matrix_to_euler_angles(diff_ori_mat, "ZYX")

        cost_pose = torch.norm(diff_pose, p=2, dim=-1, keepdim=False)
        cost_orientation = torch.norm(diff_orientation, p=2, dim=-1, keepdim=False)

        stage_cost = self.stage_pose_weight * cost_pose + self.stage_orientation_weight * cost_orientation
        stage_cost = stage_cost * self.gamma_horizon_gpu

        stage_cost = torch.sum(stage_cost, dim=1)
        return stage_cost


    def compute_terminal_cost(self, eefTraj: torch.Tensor, target_pose: Pose) -> torch.Tensor:
        ee_sample_pose = eefTraj[:,-1,0:3,3].clone()
        ee_sample_orientation = eefTraj[:,-1,0:3,0:3].clone()

        diff_pose = ee_sample_pose - target_pose.pose.to(**self.tensor_args)
        target_pose_ori_mat = euler_angles_to_matrix(target_pose.rpy.to(**self.tensor_args), "ZYX")

        diff_ori_mat = torch.matmul(torch.linalg.inv(ee_sample_orientation), target_pose_ori_mat)
        diff_orientation = matrix_to_euler_angles(diff_ori_mat, "ZYX")

        cost_pose = torch.norm(diff_pose, p=2, dim=-1, keepdim=False)
        cost_orientation = torch.norm(diff_orientation, p=2, dim=-1, keepdim=False)

        terminal_cost = self.terminal_pose_weight * cost_pose + self.terminal_orientation_weight * cost_orientation
        terminal_cost = (self.gamma **self.n_horizon) * terminal_cost

        return terminal_cost
   

    def compute_prev_stage_cost(self, eefTraj: torch.Tensor, target_pose: Pose) -> torch.Tensor:
        ee_sample_pose = eefTraj[:,0:3,3]
        ee_sample_orientation = eefTraj[:,0:3,0:3]

        diff_pose = ee_sample_pose - target_pose.pose.cpu()

        # ee_sample_quat = matrix_to_quaternion(ee_sample_orientation)

        # ee_quat_inv = quaternion_invert(ee_sample_quat)
        # q_diff = quaternion_multiply(target_pose.orientation.cpu(), ee_quat_inv) 
        # diff_orientation = quaternion_to_axis_angle(q_diff)
        diff_orientation = matrix_to_euler_angles(ee_sample_orientation, "ZYX") - target_pose.rpy.cpu()

        cost_pose = torch.norm(diff_pose, p=2, dim=-1, keepdim=False)
        cost_orientation = torch.norm(diff_orientation, p=2, dim=-1, keepdim=False)

        stage_cost = self.stage_pose_weight * cost_pose + self.stage_orientation_weight * cost_orientation
        stage_cost = stage_cost * self.gamma_horizon_cpu

        return stage_cost


    def compute_prev_terminal_cost(self, eefTraj: torch.Tensor, target_pose: Pose) -> torch.Tensor:
        ee_terminal_pose = eefTraj[-1,0:3,3]
        ee_terminal_orientation = eefTraj[-1,0:3,0:3]

        diff_pose = ee_terminal_pose - target_pose.pose.cpu()
        diff_orientation = matrix_to_euler_angles(ee_terminal_orientation, "ZYX") - target_pose.rpy.cpu()

        cost_pose = torch.norm(diff_pose, p=2, dim=-1, keepdim=False)
        cost_orientation = torch.norm(diff_orientation, p=2, dim=-1, keepdim=False)

        terminal_cost = self.terminal_pose_weight * cost_pose + self.terminal_orientation_weight * cost_orientation
        terminal_cost = (self.gamma ** self.n_horizon) * terminal_cost

        return terminal_cost
    