import torch
from rclpy.logging import get_logger

class ActionCost:
    def __init__(self, params, gamma, n_horizon, tensor_args):
        self.logger = get_logger("Action_Cost")
        self.tensor_args = tensor_args
        self.n_horizon = n_horizon
        self.gamma = gamma
        self.gamma_horizon_gpu = self.gamma ** torch.arange(self.n_horizon, **self.tensor_args)

        self.action_weight = params['weight']


    def compute_action_cost(self, uSample):
        cost_action = torch.sum(torch.pow(uSample, 2), dim=2)
        cost_action = self.action_weight * cost_action

        cost_action = cost_action * self.gamma_horizon_gpu

        cost_action = torch.sum(cost_action, dim=1)
        return cost_action
    

    def compute_prev_action_cost(self, uSample):
        cost_action = torch.sum(torch.pow(uSample, 2), dim=1)
        cost_action = self.action_weight * cost_action

        cost_action = cost_action * self.gamma_horizon_gpu

        cost_action = torch.sum(cost_action, dim=0)
        return cost_action
    