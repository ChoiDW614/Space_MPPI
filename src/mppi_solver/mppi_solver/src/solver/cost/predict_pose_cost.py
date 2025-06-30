import torch
import torch.nn as nn
import numpy as np

from mppi_solver.src.utils.pose import Pose
from mppi_solver.src.utils.rotation_conversions import euler_angles_to_matrix, matrix_to_euler_angles, matrix_to_quaternion

from rclpy.logging import get_logger

class PoseCost():
    def __init__(self, n_horizen, device):
        self._tracking_pose_weight = 3.0
        self._tracking_orientation_weight = 0.5
        self._action_weight = 10.0
        self._centering_weight = 300.0

        self._terminal_pose_weight = 10.0
        self._terminal_orientation_weight = 1.0
        
        self._gamma = 0.95
        self.n_horizen = n_horizen
        self.device = device

        self.logger = get_logger("Pose_Calculator")

        # Franka
        self.qCenter =torch.tensor([0.0, 0.0, 0.0, (-3.0718-0.0698)/2, 0.0, (3.7525-0.0175)/2, 0.0], device = self.device)


    def tracking_cost(self, eefTraj, target_pose: Pose):
        ee_sample_pose = eefTraj[:,:,0:3,3]
        ee_sample_orientation = eefTraj[:,:,0:3,0:3]
        
        diff_pose = ee_sample_pose - target_pose.pose.to(device=self.device)
        target_pose_ori_mat = euler_angles_to_matrix(target_pose.rpy.to(device=self.device), "ZYX")
        diff_ori_mat = torch.matmul(torch.linalg.inv(ee_sample_orientation), target_pose_ori_mat)
        diff_ori_quat = matrix_to_quaternion(diff_ori_mat)
        default_quat = torch.tensor([0.0,0.0,0.0,1.0],device = self.device)
        quat_error = diff_ori_quat - default_quat

        cost_pose = torch.sum(torch.pow(diff_pose, 2), dim=2)
        cost_orientation = torch.sum(torch.pow(quat_error, 2), dim=2)

        tracking_cost = self._tracking_pose_weight * cost_pose + self._tracking_orientation_weight * cost_orientation
        gamma = self._gamma ** torch.arange(self.n_horizen, device=self.device)
        tracking_cost = tracking_cost * gamma
        return tracking_cost


    def terminal_cost(self, eefTraj, target_pose: Pose):
        ee_terminal_pose = eefTraj[:,-1,0:3,3]
        ee_terminal_orientation = eefTraj[:,-1,0:3,0:3]

        diff_pose = ee_terminal_pose - target_pose.pose.to(device=self.device)
        target_pose_ori_mat = euler_angles_to_matrix(target_pose.rpy.to(device=self.device), "ZYX")
        diff_ori_mat = torch.matmul(torch.linalg.inv(ee_terminal_orientation), target_pose_ori_mat)        
        diff_ori_quat = matrix_to_quaternion(diff_ori_mat)
        default_quat = torch.tensor([0.0,0.0,0.0,1.0],device = self.device)
        quat_error = diff_ori_quat - default_quat

        cost_pose = torch.sum(torch.pow(diff_pose, 2), dim=1)
        cost_orientation = torch.sum(torch.pow(quat_error,2), dim=1)

        terminal_cost = self._terminal_pose_weight * cost_pose * 10000000 + self._terminal_orientation_weight * cost_orientation * 500000
        return terminal_cost
    
    def min_action_cost(self, uSample):
        cost_action = torch.sum(torch.pow(uSample, 2), dim=2)
        gamma = self._gamma ** torch.arange(self.n_horizen, device=self.device)
        cost_action = self._action_weight * cost_action
        return cost_action
    
    def centering_cost(self, qSample):
        cost_center = torch.sum(torch.pow(qSample-self.qCenter, 2), dim=2)
        gamma = self._gamma ** torch.arange(self.n_horizen, device=self.device)
        cost_center = self._centering_weight * cost_center
        return cost_center
    


