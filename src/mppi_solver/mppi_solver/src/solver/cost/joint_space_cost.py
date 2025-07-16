import torch

from rclpy.logging import get_logger

class JointSpaceCost:
    def __init__(self, params, gamma: float, n_horizon: int, tensor_args):
        self.logger = get_logger("Joint_Space_Cost")
        self.tensor_args = tensor_args
        self.n_horizon = n_horizon
        
        self.centering_weight = params['centering_weight']
        self.joint_traj_weight = params['tracking_weight']
        self.gamma = gamma

        self.qCenter =torch.tensor([0.0, 0.0, 0.0, (-3.0718-0.0698)/2, 0.0, (3.7525-0.0175)/2, 0.0], **self.tensor_args)


    def compute_centering_cost(self, qSample: torch.Tensor) -> torch.Tensor:
        cost_center = torch.norm(qSample-self.qCenter, p=2, dim=2)
        cost_center = self.centering_weight * cost_center

        gamma = self.gamma ** torch.arange(self.n_horizon, **self.tensor_args)
        cost_center = cost_center * gamma

        cost_center = torch.sum(cost_center, dim=1)
        return cost_center
    

    def compute_jointTraj_cost(self, qSample: torch.Tensor, jointTraj: torch.Tensor) -> torch.Tensor:
        jointTraj = jointTraj.clone().unsqueeze(0).to(**self.tensor_args)
        cost_tracking = torch.norm(qSample - jointTraj, p=2, dim=2)
        cost_tracking = self.joint_traj_weight * cost_tracking

        gamma = self.gamma ** torch.arange(self.n_horizon, **self.tensor_args)
        cost_tracking = cost_tracking * gamma

        cost_tracking = torch.sum(cost_tracking, dim=1)
        return cost_tracking
    

    def compute_prev_centering_cost(self, qSample: torch.Tensor) -> torch.Tensor:
        cost_center = torch.norm(qSample-self.qCenter, p=2, dim=1)
        cost_center = self.centering_weight * cost_center

        cost_center = cost_center
        cost_center = torch.sum(cost_center, dim=0)
        return cost_center
    
        
    def compute_prev_jointTraj_cost(self, q: torch.Tensor, jointTraj: torch.Tensor) -> torch.Tensor:
        jointTraj = jointTraj.clone().to(**self.tensor_args)
        cost_tracking = torch.norm(q - jointTraj, p=2, dim=1)
        cost_tracking = self.joint_traj_weight * cost_tracking

        cost_tracking = torch.sum(cost_tracking, dim=0)
        return cost_tracking