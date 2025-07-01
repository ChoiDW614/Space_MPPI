import torch
from rclpy.logging import get_logger

class ActionCost:
    def __init__(self, params, gamma, n_horizon, device):
        self.logger = get_logger("Action_Cost")
        self.device = device

        self.n_horizon = n_horizon

        self.action_weight = params['weight']
        self.gamma = gamma


    def compute_action_cost(self, uSample):
        cost_action = torch.sum(torch.pow(uSample, 2), dim=2)
        cost_action = self.action_weight * cost_action

        gamma = self.gamma ** torch.arange(self.n_horizon, device=self.device)
        cost_action = cost_action * gamma

        cost_action = torch.sum(cost_action, dim=1)
        return cost_action
    

    def compute_prev_action_cost(self, uSample):
        cost_action = torch.sum(torch.pow(uSample, 2), dim=1)
        cost_action = self.action_weight * cost_action

        gamma = self.gamma ** torch.arange(self.n_horizon, device=self.device)
        cost_action = cost_action * gamma

        cost_action = torch.sum(cost_action, dim=0)
        return cost_action
    