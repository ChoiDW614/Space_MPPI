import torch
from rclpy.logging import get_logger

class ActionCost:
    def __init__(self, action_weight, gamma, n_horizon, device):
        self.logger = get_logger("Action_Cost")
        self.device = device

        self.n_horizon = n_horizon

        self._action_weight = action_weight
        self._gamma = gamma


    def compute_action_cost(self, uSample):
        cost_action = torch.sum(torch.pow(uSample, 2), dim=2)
        gamma = self._gamma ** torch.arange(self.n_horizon, device=self.device)
        cost_action = self._action_weight * cost_action

        return cost_action
    

    def compute_prev_action_cost(self, uSample):
        cost_action = torch.sum(torch.pow(uSample, 2), dim=1)
        gamma = self._gamma ** torch.arange(self.n_horizon, device=self.device)
        cost_action = self._action_weight * cost_action

        return cost_action
    