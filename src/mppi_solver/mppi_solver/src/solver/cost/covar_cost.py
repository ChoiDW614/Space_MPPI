import torch


from rclpy.logging import get_logger

class CovarCost:
    def __init__(self, params, _lambda, alpha, tensor_args):
        self.logger = get_logger("Covar_Cost")
        self.tensor_args = tensor_args
        
        # Parameter
        self._lambda = _lambda
        self.alpha = alpha
        self.param_gamma = self._lambda * (1.0 - self.alpha)  # constant parameter of mppi

        # Weight
        self.covar_weight = params['weight']


        # Chan
        self.w_running = 0.01
        self.w_terminal = 10.0


    def compute_covar_cost(self, sigma_matrix, u, v):
        Sigma_inv = torch.linalg.inv(sigma_matrix).to(**self.tensor_args)
        quad_term = torch.matmul(u.unsqueeze(-2), Sigma_inv @ v.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # (batch_size, time_step)
        S = self.covar_weight * self.param_gamma * quad_term.sum(dim=1)  # (batch_size,)

        return S
    

    def compute_prev_covar_cost(self, sigma_matrix, u, noise):
        v = u + noise
        Sigma_inv = torch.linalg.inv(sigma_matrix).to(**self.tensor_args)
        quad_term = torch.matmul(u.unsqueeze(-2), Sigma_inv @ v.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # (batch_size, time_step)
        S = self.covar_weight * self.param_gamma * quad_term.sum(dim=1)  # (batch_size,)

        return S

    def compute(self, noise, sigma_matrix):
        std = torch.diagonal(sigma_matrix, dim1=-1, dim2=-2) # N, T, A
        var_inv = 1/(std**2)
        cost = self.w_running * torch.sum((noise ** 2) * var_inv, dim=-1).sum(dim=-1)
        return cost
        