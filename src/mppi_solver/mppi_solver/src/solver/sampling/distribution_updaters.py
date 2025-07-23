# distribution_updaters.py
from abc import ABC, abstractmethod
from typing import Tuple
import torch

class DistributionUpdater(ABC):
    @abstractmethod
    def update(self,
               sigma: torch.Tensor,
               init_sigma: torch.Tensor,
               kappa_eye: torch.Tensor,
               u: torch.Tensor,
               v: torch.Tensor,
               w: torch.Tensor,
               noise: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            new_sigma, new_sigma_matrix
        """
        pass


class StandardUpdater(DistributionUpdater):
    def __init__(self, n_sample: int, n_horizon: int, step_size_cov: float):
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
        self.step_size_cov = step_size_cov
        self.n_sample = n_sample
        self.n_horizon = n_horizon


    def update(self, sigma: torch.Tensor, init_sigma: torch.Tensor, kappa_eye: torch.Tensor,
                     u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, noise: torch.Tensor):
        w = w.view(-1, 1, 1)
        delta = u - noise.unsqueeze(0)
        weighted_delta = w * (delta ** 2)
        cov_update = torch.mean(torch.sum(weighted_delta, dim=0), dim=0)

        cov_update_mat = torch.diag(cov_update)
        new_sigma = (1 - self.step_size_cov) * sigma + self.step_size_cov * cov_update_mat

        if torch.isnan(new_sigma).any().item():
            new_sigma = init_sigma.clone()

        new_sigma += kappa_eye
        new_sigma_matrix = new_sigma.expand(self.n_sample, self.n_horizon, -1, -1)
        return new_sigma.clone(), new_sigma_matrix.clone()


class CMAESUpdater(DistributionUpdater):
    def __init__(self, n_sample: int, n_horizon: int, n_action: int, tensor_args):
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
        self.tensor_args = tensor_args
        self.n_sample = n_sample
        self.n_horizon = n_horizon
        self.n_action = n_action

        # CMA-ES Parameter
        self.mean  = torch.zeros((self.n_action), **self.tensor_args)
        self.ps = torch.zeros((self.n_action), **self.tensor_args)  # Evolution path
        self.pc = torch.zeros((self.n_action), **self.tensor_args)  # Evolution path for covariance matrix
        self.cc = torch.tensor(4 / (self.n_action + 4))
        self.c1 = torch.tensor(2 / ((self.n_action + 1.3) ** 2))

        # CMA-ES Computation optimization variables
        self._one_minus_c1 = 1 - self.c1
        self._pc_const = self.cc * (2 - self.cc)
        self._cs_action = self.n_action + 5
        self._cmu_action = (self.n_action + 2) ** 2
        self.iteration = 0


    def update(self, sigma: torch.Tensor, init_sigma: torch.Tensor, kappa_eye: torch.Tensor,
                     u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, noise: torch.Tensor):
        self.mu_eff = (torch.sum(w, dim=0) ** 2) / (torch.sum(w ** 2, dim=0))
        self.cmu = torch.min(self._one_minus_c1, 2 * ((self.mu_eff - 2 + 1) / self.mu_eff) / (self._cmu_action + self.mu_eff))
        self.cs = torch.tensor((self.mu_eff.item() + 2) / (self._cs_action + self.mu_eff.item()))

        # Update the mean and evolution paths
        m_old = self.mean.clone()
        self.mean = u[0].clone()
        v = v[:,0,:]

        inv_sqrt_diag = 1.0 / torch.sqrt(torch.diag(sigma))

        y = (self.mean - m_old) * inv_sqrt_diag

        coeff_sigma = torch.sqrt(self.cs * (2.0 - self.cs) * self.mu_eff)
        self.ps = (1 - self.cs) * self.ps + coeff_sigma * y

        h_sigma = (self.ps.norm() / torch.sqrt(1 - (1 - self.cs) ** (2 * (self.iteration + 1)))) < (1.4 + 2 / (self.n_action + 1))
        self.pc = (1 - self.cc) * self.pc + h_sigma * torch.sqrt(self._pc_const * self.mu_eff) * y

        # Update the covariance matrix
        delta_rank1 = torch.outer(self.pc, self.pc) 

        y_k = (v - m_old) * inv_sqrt_diag
        delta_rankmu = torch.einsum('i,ij,ik->jk', w, y_k, y_k)

        new_sigma = (self._one_minus_c1 - self.cmu) * sigma + self.c1 * delta_rank1 + self.cmu * delta_rankmu
        new_sigma += kappa_eye
        new_sigma_matrix = new_sigma.expand(self.n_sample, self.n_horizon, -1, -1)

        self.iteration += 1
        return new_sigma.clone(), new_sigma_matrix.clone()
