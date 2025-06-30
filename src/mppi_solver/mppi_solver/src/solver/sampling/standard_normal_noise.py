import torch

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
        self.sigma = torch.eye((self.n_action), device = self.device) * 0.3

        self.sigma_matrix : torch.Tensor


    def sampling(self):
        standard_normal_noise = torch.randn(self.n_sample, self.n_horizon, self.n_action, device=self.device)
        self.sigma_matrix = self.sigma.expand(self.n_sample, self.n_horizon, -1, -1)
        noise = torch.matmul(standard_normal_noise.unsqueeze(-2), self.sigma_matrix).squeeze(-2)
        return noise


    def get_sample_joint(self, samples: torch.Tensor, q: torch.Tensor, qdot: torch.Tensor, dt):
        qdot0 = qdot.unsqueeze(0).unsqueeze(0).expand(self.n_sample, 1, self.n_action)  # (n_sample, 1, n_action)
        q0 = q.unsqueeze(0).unsqueeze(0).expand(self.n_sample, 1, self.n_action)        # (n_sample, 1, n_action)
        v = torch.cumsum(samples * dt, dim=1) + qdot0  # (n_sample, n_horizon, n_action)

        # 이전 속도: [v0, v0+..., ..., v_{N-1}]
        v_prev = torch.cat([qdot0, v[:, :-1, :]], dim=1)  # (n_sample, n_horizon, n_action)

        # 누적 위치 계산: q[i] = q[i-1] + v[i-1] * dt + 0.5 * a[i] * dt^2
        dq = v_prev * dt + 0.5 * samples * dt**2
        q = torch.cumsum(dq, dim=1) + q0

        return q


    def get_prev_sample_joint(self, u_prev: torch.Tensor, q: torch.Tensor, qdot: torch.Tensor, dt):
        qdot0 = qdot.unsqueeze(0).expand(1, self.n_action)  # (1, n_action)
        q0 = q.unsqueeze(0).expand(1, self.n_action)        # (1, n_action)
        v = torch.cumsum(u_prev * dt, dim=0) + qdot0  # (n_horizon, n_action)

        v_prev = torch.cat([qdot0, v[:-1, :]], dim=0)  # (n_horizon, n_action)

        dq = v_prev * dt + 0.5 * u_prev * dt**2
        q = torch.cumsum(dq, dim=0) + q0
        return q.unsqueeze(0)
    