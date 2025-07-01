import torch

from rclpy.logging import get_logger


class StandardSamplling:
    def __init__(self, params, device):
        self.logger = get_logger("Standard_Sampling")

        # Torch GPU
        self.device = device

        # Sampling Parameter
        self.n_sample  = params['mppi']['samples']
        self.n_horizon = params['mppi']['horizon']
        self.n_action  = params['mppi']['action']
        self.sigma_scale = params['mppi']['sigma_scale']
        self.sigma = torch.eye((self.n_action), device = self.device) * self.sigma_scale
        self.init_sigma = self.sigma.clone()

        self.sigma_matrix = self.sigma.expand(self.n_sample, self.n_horizon, -1, -1)

        # Update Parameter
        self.step_size_cov  = 0.05
        self.kappa = 0.005


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
    

    def update_distribution(self, actions: torch.Tensor, weights: torch.Tensor, noise: torch.Tensor):
        delta = actions - noise.unsqueeze(0)
        weighted_delta = weights * (delta ** 2)
        cov_update = torch.mean(torch.sum(weighted_delta, dim=0), dim=0)

        cov_update_mat = torch.diag(cov_update)
        self.sigma = (1 - self.step_size_cov) * self.sigma + self.step_size_cov * cov_update_mat
        self.sigma += self.kappa * torch.eye(self.n_action, device=self.device)
        self.sigma_matrix = self.sigma.expand(self.n_sample, self.n_horizon, -1, -1)

        self.logger.info(f"Updated Sigma: {torch.diag(self.sigma)}")
