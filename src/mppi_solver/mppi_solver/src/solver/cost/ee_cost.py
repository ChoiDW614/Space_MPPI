import torch
from rclpy.logging import get_logger

class EECost:
    def __init__(self, params, gamma, n_horizon, dt, tensor_args):
        self.logger = get_logger("EE_Cost")
        self.tensor_args = tensor_args

        self.n_horizon = n_horizon
        self.gamma = gamma
        self.dt = dt

        self.ee_weight = params['weight']
        self.distance_limit = params['distance_limit']


    def compute_ee_cost(self, uSample: torch.Tensor, v_prev:torch.Tensor, jacobian: torch.Tensor, target_dist: torch.Tensor):
        if target_dist < self.distance_limit:
            joint_vel = v_prev[0].unsqueeze(0) + torch.cumsum(uSample * self.dt, dim=1)
            jacobian = jacobian.to(**self.tensor_args)

            ee_vel = torch.einsum('shja,sha->shj', jacobian, joint_vel)
            cost_ee = self.ee_weight * torch.norm(ee_vel, p=2, dim=2)

            gamma = self.gamma ** torch.arange(self.n_horizon, **self.tensor_args)
            cost_ee = cost_ee * gamma

            cost_ee = torch.sum(cost_ee, dim=1)
        else:
            cost_ee = torch.zeros((uSample.shape[0]), **self.tensor_args)

        return cost_ee
    

    def compute_prev_ee_cost(self, uSample: torch.Tensor, v_prev:torch.Tensor, jacobian: torch.Tensor, target_dist: torch.Tensor):
        if target_dist < self.distance_limit:
            joint_vel = v_prev[0].unsqueeze(0) + torch.cumsum(uSample * self.dt, dim=0)
            jacobian = jacobian.to(**self.tensor_args)

            ee_vel = torch.einsum('hja,ha->hj', jacobian, joint_vel)
            cost_ee = self.ee_weight * torch.norm(ee_vel, p=2, dim=1)

            gamma = self.gamma ** torch.arange(self.n_horizon, **self.tensor_args)
            cost_ee = cost_ee * gamma

            cost_ee = torch.sum(cost_ee, dim=0)
        else:
            cost_ee = torch.zeros((0), **self.tensor_args)

        return cost_ee
