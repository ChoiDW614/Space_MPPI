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
        ee_sample_pose = eefTraj[:,:-1,0:3,3]
        ee_sample_orientation = eefTraj[:,:-1,0:3,0:3]

        diff_pose = ee_sample_pose - target_pose.pose.to(**self.tensor_args)
        cost_pose = torch.norm(diff_pose, p=2, dim=-1, keepdim=False)

        target_pose_quat = target_pose.orientation.to(**self.tensor_args)
        diff_ori = matrix_to_quaternion(ee_sample_orientation) * target_pose_quat
        self.logger.info(f"target_pose_quat : {target_pose_quat}")
        self.logger.info(f"ee_pose_quat : {matrix_to_quaternion(ee_sample_orientation)}")
        
        cost_orientation = torch.norm(diff_ori, p=2, dim=-1, keepdim=False)
        cost_orientation = (1.0 - torch.pow(cost_orientation, 2))

        stage_cost = self.stage_pose_weight * cost_pose + self.stage_orientation_weight * cost_orientation
        stage_cost = stage_cost * self.gamma_horizon_gpu

        stage_cost = torch.sum(stage_cost, dim=1)
        return stage_cost


    def compute_terminal_cost(self, eefTraj: torch.Tensor, target_pose: Pose) -> torch.Tensor:
        ee_sample_pose = eefTraj[:,-1,0:3,3].clone()
        ee_sample_orientation = eefTraj[:,-1,0:3,0:3].clone()

        diff_pose = ee_sample_pose - target_pose.pose.to(**self.tensor_args)
        cost_pose = torch.norm(diff_pose, p=2, dim=-1, keepdim=False)

        target_pose_quat = target_pose.orientation.to(**self.tensor_args)
        diff_ori = matrix_to_quaternion(ee_sample_orientation) * target_pose_quat

        cost_orientation = torch.norm(diff_ori, p=2, dim=-1, keepdim=False)
        cost_orientation = (1.0 - torch.pow(cost_orientation, 2))

        terminal_cost = self.terminal_pose_weight * cost_pose + self.terminal_orientation_weight * cost_orientation
        terminal_cost = (self.gamma **self.n_horizon) * terminal_cost
        return terminal_cost


    def compute_prev_stage_cost(self, eefTraj: torch.Tensor, target_pose: Pose) -> torch.Tensor:
        ee_sample_pose = eefTraj[:,0:3,3]
        ee_sample_orientation = eefTraj[:,0:3,0:3]

        diff_pose = ee_sample_pose - target_pose.pose.cpu()
        cost_pose = torch.norm(diff_pose, p=2, dim=-1, keepdim=False)

        target_pose_quat = target_pose.orientation.cpu()
        diff_ori = matrix_to_quaternion(ee_sample_orientation) * target_pose_quat

        cost_orientation = torch.norm(diff_ori, p=2, dim=-1, keepdim=False)
        cost_orientation = (1.0 - torch.pow(cost_orientation, 2))

        stage_cost = self.stage_pose_weight * cost_pose + self.stage_orientation_weight * cost_orientation
        stage_cost = stage_cost * self.gamma_horizon_cpu

        return stage_cost


    def compute_prev_terminal_cost(self, eefTraj: torch.Tensor, target_pose: Pose) -> torch.Tensor:
        ee_terminal_pose = eefTraj[-1,0:3,3]
        ee_terminal_orientation = eefTraj[-1,0:3,0:3]

        diff_pose = ee_terminal_pose - target_pose.pose.cpu()
        cost_pose = torch.norm(diff_pose, p=2, dim=-1, keepdim=False)

        target_pose_quat = target_pose.orientation.cpu()
        diff_ori = matrix_to_quaternion(ee_terminal_orientation) * target_pose_quat

        cost_orientation = torch.norm(diff_ori, p=2, dim=-1, keepdim=False)
        cost_orientation = (1.0 - torch.pow(cost_orientation, 2))

        terminal_cost = self.terminal_pose_weight * cost_pose + self.terminal_orientation_weight * cost_orientation
        terminal_cost = (self.gamma ** self.n_horizon) * terminal_cost

        return terminal_cost
    
    def compute(self,T, goal):
        pos_ee = T[..., :3, 3] # B,T,3
        pos_goal = goal[:3, 3]  # 3,
        pos_diff = pos_ee - pos_goal.unsqueeze(0).unsqueeze(0) # B,T,3
        pos_cost = (pos_diff ** 2).sum(dim=-1) # B.T



        R_ee = T[..., :3, :3] # B,T,3,3
        R_goal = goal[:3, :3].expand_as(R_ee)

        q_ee = self.rotmat_to_quat(R_ee)
        q_goal = self.rotmat_to_quat(R_goal)

        q_ee = q_ee / (q_ee.norm(dim=-1, keepdim=True) + 1e-8)
        q_goal = q_goal / (q_goal.norm(dim=-1, keepdim=True) + 1e-8)
        dot = torch.sum(q_ee * q_goal, dim=-1)
        rot_cost = (1.0 - dot**2) # B,T

        running_pos_cost = pos_cost[:, :-1] * self.stage_pose_weight
        running_rot_cost = rot_cost[:, :-1] * self.stage_orientation_weight

        terminal_pos_cost = pos_cost[:, -1] * self.terminal_pose_weight
        terminal_rot_cost = rot_cost[:, -1] * self.terminal_orientation_weight

        cost = running_pos_cost.sum(dim=-1) + running_rot_cost.sum(dim=-1) + terminal_pos_cost + terminal_rot_cost  # (B,)
        return cost

    def rotmat_to_quat(self, R):

        m00 = R[..., 0, 0]
        m01 = R[..., 0, 1]
        m02 = R[..., 0, 2]
        m10 = R[..., 1, 0]
        m11 = R[..., 1, 1]
        m12 = R[..., 1, 2]
        m20 = R[..., 2, 0]
        m21 = R[..., 2, 1]
        m22 = R[..., 2, 2]

        trace = m00 + m11 + m22
        qw = 0.5 * torch.sqrt(torch.clamp(trace + 1.0, min=1e-6))
        qx = 0.5 * torch.sign(m21 - m12) * torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=1e-6))
        qy = 0.5 * torch.sign(m02 - m20) * torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=1e-6))
        qz = 0.5 * torch.sign(m10 - m01) * torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=1e-6))

        q = torch.stack([qx, qy, qz, qw], dim=-1)  # (..., 4)
        return q