import math
import torch
import torch.nn as nn
import numpy as np

from mppi_solver.src.utils.pose import Pose
from mppi_solver.src.utils.rotation_conversions import euler_angles_to_matrix, matrix_to_euler_angles, matrix_to_quaternion

from rclpy.logging import get_logger

class DistCost():
    def __init__(self, n_action, device):
        self.logger = get_logger("DistCost")
        self.device = device
        self.n_action = n_action
        self.disp_weight = torch.ones((n_action), device=self.device)
        self.dist_weight = 10.0
        # position_gaussian_params: {'n':0, 'c':0.0, 's':0.0, 'r':10.0}
        # orientation_gaussian_params: {'n':0, 'c':0.0, 's':0.0, 'r':10.0}

        # Gaussian parameters for position and orientation
        self.omega = {'n':0,'c':0,'s':0,'r':10.0}
        self._wn = self.omega['n']
        self._wc = self.omega['c']
        self._ws = self.omega['s']
        self._wr = self.omega['r']

        if(len(self.omega.keys()) > 0):
            self.n_pow = math.pow(-1.0, self.omega['n'])


    def dist_cost(self, cur_states, goal_states):
        disp_states = cur_states - goal_states
        weighted_disp_states = torch.einsum('k,ijk->ijk', self.disp_weight, disp_states)
        dist = torch.norm(weighted_disp_states, p=2, dim=-1,keepdim=False)

        cost = self.dist_weight * self.gaussian_projection(dist)
        return cost


    def gaussian_projection(self, cost_value):
        if(self._wc == 0.0):
            return cost_value
        exp_term = torch.div(-1.0 * (cost_value - self._ws)**2, 2.0 * (self._wc**2))

        cost = 1.0 - self.n_pow * torch.exp(exp_term) + self._wr * torch.pow(cost_value - self._ws, 4)
        return cost
    