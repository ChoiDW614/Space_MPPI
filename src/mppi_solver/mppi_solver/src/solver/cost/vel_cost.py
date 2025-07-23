import math
import torch
import torch.nn as nn
import numpy as np

from mppi_solver.src.utils.pose import Pose
from mppi_solver.src.utils.rotation_conversions import euler_angles_to_matrix, matrix_to_euler_angles, matrix_to_quaternion

from rclpy.logging import get_logger

class VelCost():
    def __init__(self, tensor_args):
        self.logger = get_logger("VelCost")
        self.tensor_args = tensor_args

        self.terminal_vel_weight = 1000.0

    def compute_terminal_vel_cost(self, vTraj):
        v_T = vTraj[:,-1,:].clone()
        cost = self.terminal_vel_weight * torch.linalg.norm(v_T,dim=-1)

        return cost


    def compute_base_cost(self, jacob_bm, v_sample):
        v_base = torch.einsum('ntij,ntj->nti', jacob_bm, v_sample)
        v_base = torch.linalg.norm(v_base, dim = -1)

        cost = (100000 * v_base).sum(dim=1)
        return cost