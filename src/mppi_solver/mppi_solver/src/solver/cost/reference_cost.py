import torch
from rclpy.logging import get_logger

class ReferenceCost:
    def __init__(self, params, gamma, n_horizon, dt, tensor_args):
        self.logger = get_logger("Reference_Cost")
        self.tensor_args = tensor_args

        self.n_horizon = n_horizon
        self.gamma = gamma
        self.dt = dt

        self.reference_weight = params['weight']


    def compute_reference_cost(self, ee_joint: torch.Tensor, pose_trajectories:torch.Tensor):
        pose_trajectories = pose_trajectories.unsqueeze(0).to(**self.tensor_args)
        reference_diff = ee_joint - pose_trajectories

        cost_reference = self.reference_weight * torch.norm(reference_diff, p='fro', dim=(2, 3))

        gamma = self.gamma ** torch.arange(self.n_horizon, **self.tensor_args)
        cost_reference = cost_reference * gamma

        cost_reference = torch.sum(cost_reference, dim=1)
        return cost_reference
    

    def compute_prev_reference_cost(self, ee_joint: torch.Tensor, pose_trajectories:torch.Tensor):
        pose_trajectories = pose_trajectories.to(**self.tensor_args)
        reference_diff = ee_joint.squeeze(0) - pose_trajectories
        cost_reference = self.reference_weight * torch.norm(reference_diff, p='fro', dim=(1, 2))

        gamma = self.gamma ** torch.arange(self.n_horizon, **self.tensor_args)
        cost_reference = cost_reference * gamma

        cost_reference = torch.sum(cost_reference, dim=0)

        return cost_reference
