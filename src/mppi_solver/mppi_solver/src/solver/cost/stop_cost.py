import torch
from rclpy.logging import get_logger

class StopCost:
    def __init__(self, params, gamma, n_horizon, dt, tensor_args):
        self.logger = get_logger("Stop_Cost")
        self.tensor_args = tensor_args

        self.n_horizon = n_horizon
        self.gamma = gamma
        self.dt = dt

        self.stop_weight = params['weight']
        self.v_max = params['v_max']
        self.vmax = torch.tensor(self.v_max, **tensor_args).unsqueeze(0)


    def compute_stop_cost(self, uSample: torch.Tensor, v_prev: torch.Tensor):
        vel = v_prev[0].unsqueeze(0) + torch.cumsum(uSample * self.dt, dim=1)
        abs_zero_vel = torch.clamp_min(torch.abs(vel) - self.vmax, min=0.0)

        cost_stop = self.stop_weight * torch.norm(abs_zero_vel, p=2, dim=2)

        gamma = self.gamma ** torch.arange(self.n_horizon, **self.tensor_args)
        cost_stop = cost_stop * gamma

        cost_stop = torch.sum(cost_stop, dim=1)
        return cost_stop
    

    def compute_prev_stop_cost(self, uSample: torch.Tensor, v_prev: torch.Tensor):
        vel = v_prev[0].unsqueeze(0) + torch.cumsum(uSample * self.dt, dim=0)
        abs_zero_vel = torch.clamp_min(torch.abs(vel) - self.vmax, min=0.0)

        cost_stop = self.stop_weight * torch.norm(abs_zero_vel, p=2, dim=1)

        gamma = self.gamma ** torch.arange(self.n_horizon, **self.tensor_args)
        cost_stop = cost_stop * gamma

        cost_stop = torch.sum(cost_stop, dim=0)
        return cost_stop