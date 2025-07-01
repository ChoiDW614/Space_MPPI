import torch

from rclpy.logging import get_logger


class StandardSampling:
    def __init__(self, params, device):
        self.logger = get_logger("Standard_Sampling")

        # Torch GPU
        self.device = device

        # Sampling Parameter
        self.n_sample  = params['mppi']['sample']
        self.n_horizon = params['mppi']['horizon']
        self.n_action  = params['mppi']['action']
        self.sigma_scale: float = params['sample']['sigma_scale']
        self.sigma: torch.Tensor = torch.eye((self.n_action), device = self.device) * self.sigma_scale
        self.init_sigma: torch.Tensor = self.sigma.clone()

        self.sigma_matrix = self.sigma.expand(self.n_sample, self.n_horizon, -1, -1)

        # Update Parameter
        self.sigma_update = params['sample']['sigma_update']

        if self.sigma_update:
            self.kappa = params['sample']['kappa']
            self.kappa_eye = self.kappa * torch.eye((self.n_action), device=self.device)

            if params['sample']['sigma_update_type'] == "CMA_ES":
                self.update_fn = self.update_distribution_CMA_ES
                # CMA-ES Parameter
                self.mean  = torch.zeros((self.n_action), device=self.device)
                self.ps = torch.zeros((self.n_action), device=self.device)  # Evolution path
                self.pc = torch.zeros((self.n_action), device=self.device)  # Evolution path for covariance matrix
                self.cc = torch.tensor(4 / (self.n_action + 4))
                self.c1 = torch.tensor(2 / ((self.n_action + 1.3) ** 2))
                # CMA-ES Computation optimization variables
                self.kappa_eye = self.kappa * torch.eye((self.n_action), device=self.device)
                self._one_minus_c1 = 1 - self.c1
                self._cs_action = self.n_action + 5
                self._cmu_action = (self.n_action + 2) ** 2
                self._pc_const = self.cc * (2 - self.cc)
                self.iteration = 0
            if params['sample']['sigma_update_type'] == "standard":
                self.update_fn = self.update_distribution_standard
                # Standard Parameter
                self.step_size_cov = params['sample']['step_size_cov']
        return


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
    
    
    def update_distribution(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, noise: torch.Tensor):
        if self.sigma_update:
            self.update_fn(u, v, w, noise)
        return
    

    def update_distribution_standard(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, noise: torch.Tensor):
        """Perform the covariance adaptation step of the MPPI controller.

        This function implements the distribution update for Model Predictive
        Path Integral (MPPI) control, comprising the following operations:
        1. Compute per-sample deviations delta = u - noise.
        2. Calculate weighted diagonal covariance update from squared deviations.
        3. Blend the new estimate with the current covariance via step_size_cov.
        4. Reset to initial covariance if NaNs are detected.
        5. Regularize by adding kappa*I.
        6. Expand the covariance across samples and the time horizon.
        7. Log the updated diagonal entries for diagnostics.

        Args:
            u (torch.Tensor): Sampled control sequences, shape (n_sample, n_action).
            v (torch.Tensor): (Unused) Placeholder for sampled trajectories.
            w (torch.Tensor): Importance-sampling weights, shape (n_sample,).
            noise (torch.Tensor): Noise offsets used to generate u, shape (n_sample, n_action).

        Returns:
            None: Updates self.sigma and self.sigma_matrix in place.
        """
        delta = u - noise.unsqueeze(0)
        weighted_delta = w * (delta ** 2)
        cov_update = torch.mean(torch.sum(weighted_delta, dim=0), dim=0)

        cov_update_mat = torch.diag(cov_update)
        self.sigma = (1 - self.step_size_cov) * self.sigma + self.step_size_cov * cov_update_mat

        if torch.isnan(self.sigma).any().item():
            self.sigma = self.init_sigma.clone()

        self.sigma += self.kappa_eye
        self.sigma_matrix = self.sigma.expand(self.n_sample, self.n_horizon, -1, -1)

        self.logger.info(f"Updated Sigma: {torch.diag(self.sigma)}")
        return


    def update_distribution_CMA_ES(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, noise: torch.Tensor):
        """Update the CMA-ES distribution parameters (mean, evolution paths, and covariance matrix).

        This method implements the distribution update step of the Covariance Matrix
        Adaptation Evolution Strategy (CMA-ES). It performs the following operations:
        1. Computes the effective sample size (μ_eff) from the weight vector w.
        2. Determines adaptation rates c_sigma, c_c, c_1, and c_μ based on μ_eff and algorithm constants.
        3. Updates the evolution paths p_sigma (step-size control) and p_c (covariance control).
        4. Constructs rank-one and rank-μ updates for the covariance matrix (Σ).
        5. Regularizes Σ with the κ-scaled identity matrix.
        6. Expands Σ into Σ_matrix for trajectory sampling.
        7. Increments the internal iteration counter.

        Args:
            u (torch.Tensor): Candidate solutions sorted by fitness,
                shape (num_samples, dimension), where u[0] is the best solution.
            v (torch.Tensor): Sampled trajectories, shape (num_samples, horizon, dimension).
            w (torch.Tensor): Weight coefficients for each candidate solution,
                shape (num_samples,).
            noise (torch.Tensor): Noise samples for exploration (unused in this step).

        Returns:
            None: Updates the internal state (mean, covariance, evolution paths)
                of the CMA-ES instance in place.
        """
        # Compute effective sample size (μ_eff)
        self.mu_eff = (torch.sum(w, dim=0) ** 2) / (torch.sum(w ** 2, dim=0))
        self.cmu = torch.min(self._one_minus_c1, 2 * ((self.mu_eff - 2 + 1) / self.mu_eff) / (self._cmu_action + self.mu_eff))
        self.cs = torch.tensor((self.mu_eff.item() + 2) / (self._cs_action + self.mu_eff.item()))

        # Update the mean and evolution paths
        m_old = self.mean.clone()
        self.mean = u[0].clone()
        v = v[:,0,:]

        inv_sqrt_diag = 1.0 / torch.sqrt(torch.diag(self.sigma))

        y = (self.mean - m_old) * inv_sqrt_diag

        coeff_sigma = torch.sqrt(self.cs * (2.0 - self.cs) * self.mu_eff)
        self.ps = (1 - self.cs) * self.ps + coeff_sigma * y

        h_sigma = (self.ps.norm() / torch.sqrt(1 - (1 - self.cs) ** (2 * (self.iteration + 1)))) < (1.4 + 2 / (self.n_action + 1))
        self.pc = (1 - self.cc) * self.pc + h_sigma * torch.sqrt(self._pc_const * self.mu_eff) * y

        # Update the covariance matrix
        delta_rank1 = torch.outer(self.pc, self.pc) 

        y_k = (v - m_old) * inv_sqrt_diag
        delta_rankmu = torch.einsum('i,ij,ik->jk', w, y_k, y_k)

        self.sigma = (self._one_minus_c1 - self.cmu) * self.sigma + self.c1 * delta_rank1 + self.cmu * delta_rankmu
        self.sigma += self.kappa_eye
        self.sigma_matrix = self.sigma.expand(self.n_sample, self.n_horizon, -1, -1)

        self.iteration += 1
        # self.logger.info(f"CMA-ES Updated Sigma Diagonals: {torch.diag(self.sigma)}")
        return
    