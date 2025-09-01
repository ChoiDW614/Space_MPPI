import torch
import time
from rclpy.logging import get_logger


class StandardSamplling:
    def __init__(self, n_sample : int, n_horizon : int, n_action : int, device):
        self.logger = get_logger("Standard_Sampling")

        # Torch GPU
        self.device = device

        # Sampling Parameter
        self.n_sample = n_sample
        self.n_horizon = n_horizon
        self.n_action = n_action

        # Standard Dev.
        self.sigma = torch.eye((self.n_action), device = self.device) # 표준편차 0.1
        self.sigma *= 0.1
        # self.sigma[:3, :3] *= 3.0
        # self.sigma[3:, 3:] *= 3.0

        self.sigma_matrix = self.sigma.expand(self.n_sample, self.n_horizon, -1, -1)

        self.sigma_update = False
        self.init_sigma: torch.Tensor = self.sigma.clone()
        torch.manual_seed(43)


    def sampling(self):
        # standard_normal_noise = torch.randn(self.n_sample, self.n_horizon, self.n_action, device=self.device)
        standard_normal_noise = torch.randn(self.n_sample, 1, self.n_action, device=self.device)
        standard_normal_noise = standard_normal_noise.expand(-1, self.n_horizon, -1)
        self.sigma_matrix = self.sigma.expand(self.n_sample, self.n_horizon, -1, -1)
        noise = torch.matmul(standard_normal_noise.unsqueeze(-2), self.sigma_matrix).squeeze(-2)
        return noise


    def get_sample_joint(self, samples: torch.Tensor, q: torch.Tensor, qdot: torch.Tensor, dt):
        # samples : (N, T, A)
        
        # qdot0 는 (A,) 사이즈인데 얘를 unsqueeze 두번
        # qdot0 는 (1,1,A)
        # Expand (N, 1, A)
        qdot0 = qdot.unsqueeze(0).unsqueeze(0).expand(self.n_sample, 1, self.n_action)  # (n_sample, 1, n_action)
        q0 = q.unsqueeze(0).unsqueeze(0).expand(self.n_sample, 1, self.n_action)        # (n_sample, 1, n_action)
        
        # Sample * dt = dv
        v = torch.cumsum(samples * dt, dim=1) + qdot0  # (n_sample, n_horizon, n_action)
        v_prev = torch.cat([qdot0, v[:, :-1, :]], dim=1)  # (n_sample, n_horizon, n_action)
        dq = v_prev * dt + 0.5 * samples * dt**2
        # OR / v_prev 가 가속도를 통해 나온건데 굳이 더해줘야하나?
        # dq = v_prev * dt

        q = torch.cumsum(dq, dim=1) + q0

        return q, v


    def get_prev_sample_joint(self, u_prev: torch.Tensor, q: torch.Tensor, qdot: torch.Tensor, dt):
        qdot0 = qdot.unsqueeze(0).expand(1, self.n_action)  # (1, n_action)
        q0 = q.unsqueeze(0).expand(1, self.n_action)        # (1, n_action)
        v = torch.cumsum(u_prev * dt, dim=0) + qdot0  # (n_horizon, n_action)

        v_prev = torch.cat([qdot0, v[:-1, :]], dim=0)  # (n_horizon, n_action)

        dq = v_prev * dt + 0.5 * u_prev * dt**2
        q = torch.cumsum(dq, dim=0) + q0
        return q.unsqueeze(0), v.unsqueeze(0)
    
