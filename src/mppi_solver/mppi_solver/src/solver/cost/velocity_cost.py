import torch

class CostVelZero:
    def __init__(self):
        self.w_running = 5.0
        self.w_terminal = 100.0


    def compute(self, vel):
        running_vel = vel[:, :-1, :]
        terminal_vel = vel[:, -1, :]
        running_cost = self.w_running * torch.sum(running_vel ** 2, dim=-1).sum(dim=-1) 
        terminal_cost = self.w_terminal * torch.sum(terminal_vel ** 2, dim=-1) 
        cost = running_cost + terminal_cost
        return cost
        
    def compute_vel_penalty(self, vel):
        # vel : N, T, A
        vel_limit = 0.0698132
        vel_mask = torch.abs(vel) > vel_limit
        
        penalty_weight = 1e+5
        vel_mask = penalty_weight * vel_mask
        S = torch.sum(vel_mask, dim=2)
        S = torch.sum(S, dim=1)

        return S



