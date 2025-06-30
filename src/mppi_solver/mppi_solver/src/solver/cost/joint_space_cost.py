import torch

from rclpy.logging import get_logger

class JointSpaceCost:
    def __init__(self, centering_weight, joint_traj_weight, gamma, n_horizon, device):
        self.device = device
        self.n_horizon = n_horizon
        
        self.centering_weight = centering_weight
        self.joint_traj_weight = joint_traj_weight
        self.gamma = gamma

        self.qCenter =torch.tensor([0.0, 0.0, 0.0, (-3.0718-0.0698)/2, 0.0, (3.7525-0.0175)/2, 0.0], device = self.device)


    def compute_centering_cost(self, qSample):
        cost_center = torch.sum(torch.pow(qSample-self.qCenter, 2), dim=2)
        gamma = self.gamma ** torch.arange(self.n_horizon, device=self.device)
        cost_center = self.centering_weight * cost_center
        return cost_center
    

    def compute_jointTraj_cost(self, qSample, jointTraj):
        # qSample : samples by timewindow by joint
        # jointTraj : timewindow by joint
        jointTraj = jointTraj.clone().unsqueeze(0).to(device = self.device)
        cost_tracking = torch.sum(torch.pow(qSample - jointTraj, 2), dim = 2)
        gamma = self.gamma ** torch.arange(self.n_horizon, device=self.device)
        cost_tracking = self.joint_traj_weight * cost_tracking
        return cost_tracking
    

    def compute_prev_centering_cost(self, qSample):
        cost_center = torch.sum(torch.pow(qSample - self.qCenter, 2), dim=1)
        gamma = self.gamma ** torch.arange(self.n_horizon, device=self.device)
        cost_center = self.centering_weight * cost_center
        return cost_center
    
        
    def compute_prev_jointTraj_cost(self, eefTraj, jointTraj):
        jointTraj = jointTraj.clone().to(device = self.device)
        cost_tracking = torch.sum(torch.pow(eefTraj - jointTraj, 2), dim=1)
        gamma = self.gamma ** torch.arange(self.n_horizon, device=self.device)
        cost_tracking = self.joint_traj_weight * cost_tracking
        return cost_tracking