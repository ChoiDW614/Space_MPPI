import math
import torch
import torch.nn as nn
import numpy as np

from mppi_solver.src.utils.pose import Pose
from mppi_solver.src.utils.rotation_conversions import euler_angles_to_matrix, matrix_to_euler_angles, matrix_to_quaternion

from rclpy.logging import get_logger

class BaseDisturbanceCost():
    def __init__(self, params, gamma, n_horizon, tensor_args):
        self.logger = get_logger("VelCost")
        self.tensor_args = tensor_args

        self.n_horizon = n_horizon
        self.gamma = gamma
        self.gamma_horizon_gpu = self.gamma ** torch.arange(self.n_horizon, **self.tensor_args)

        self.terminal_vel_weight = 1000.0
        self.base_disturbance_weight = params['weight']

    def compute_terminal_vel_cost(self, vTraj):
        v_T = vTraj[:,-1,:].clone()
        cost = self.terminal_vel_weight * torch.linalg.norm(v_T,dim=-1)

        return cost


    def compute_base_disturbance_cost(self, jacob_bm: torch.Tensor, v_sample: torch.Tensor):
        v_base = torch.einsum('ntij,ntj->nti', jacob_bm, v_sample)
        v_base = torch.norm(v_base, p=2, dim=-1)

        cost_disturbance = self.base_disturbance_weight * v_base

        cost_disturbance = cost_disturbance * self.gamma_horizon_gpu
        cost_disturbance = torch.sum(cost_disturbance, dim=1)
        return cost_disturbance
    

    def compute_prev_base_disturbance_cost(self, jacob_bm: torch.Tensor, v_sample: torch.Tensor):
        v_base = torch.einsum('tij,tj->ti', jacob_bm.squeeze(0), v_sample)
        v_base = torch.norm(v_base, p=2, dim=-1)

        cost_disturbance = self.base_disturbance_weight * v_base

        cost_disturbance = cost_disturbance * self.gamma_horizon_gpu
        cost_disturbance = torch.sum(cost_disturbance, dim=0)
        return cost_disturbance
    
    def compute_base_disturbance(self, jacob_bm: torch.Tensor, v_sample: torch.Tensor):
        v_base = torch.einsum('tij,tj->ti', jacob_bm.squeeze(0), v_sample)
        v_base = v_base[0,:]
        return v_base
    