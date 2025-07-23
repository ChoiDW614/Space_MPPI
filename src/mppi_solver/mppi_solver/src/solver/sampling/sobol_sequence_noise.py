import torch
import time
from rclpy.logging import get_logger

from mppi_solver.src.solver.sampling.distribution_updaters import StandardUpdater, CMAESUpdater

class SobolSampling:
    def __init__(self, params, tensor_args):
        self.logger = get_logger("Sobol_Sampling")

        # Torch GPU
        self.tensor_args = tensor_args

        # Sampling Parameter
        self.n_sample  = params['mppi']['sample']
        self.n_horizon = params['mppi']['horizon']
        self.n_action  = params['mppi']['action']
        self.n_total_sample = self.n_sample * self.n_horizon * self.n_action

        self.seed = params['sample']['seed']
        if self.seed == 0:
            self.seed = time.time_ns()

        # Sobol sequence
        self.sobol_engine = torch.quasirandom.SobolEngine(dimension=3, scramble=False, seed=self.seed)

        self.sigma_scale: float = params['sample']['sigma_scale']
        self.sigma: torch.Tensor = torch.eye((self.n_action), **self.tensor_args) * self.sigma_scale
        self.init_sigma: torch.Tensor = self.sigma.clone()
        self.sigma_matrix = self.sigma.expand(self.n_sample, self.n_horizon, -1, -1)

        # Update Parameter
        self.sigma_update = params['sample']['sigma_update']

        if self.sigma_update:
            self.kappa = params['sample']['kappa']
            self.kappa_eye = self.kappa * torch.eye((self.n_action), **self.tensor_args)

            if params['sample']['sigma_update_type'] == "CMA_ES":
                self.updater = CMAESUpdater(self.n_sample, self.n_horizon, self.n_action, self.tensor_args)
            if params['sample']['sigma_update_type'] == "standard":
                # Standard Parameter
                self.step_size_cov = params['sample']['standard']['step_size_cov']
                self.updater = StandardUpdater(self.n_sample, self.n_horizon, self.step_size_cov)
        return
    

    def sampling(self):
        sobol_points = self.sobol_engine.draw(self.n_total_sample).to(**self.tensor_args)
        sobol_noise = sobol_points[:,0].reshape(self.n_sample, self.n_horizon, self.n_action)

        self.sigma_matrix = self.sigma.expand(self.n_sample, self.n_horizon, -1, -1)
        sobol_noise = torch.matmul(sobol_noise.unsqueeze(-2), self.sigma_matrix).squeeze(-2)
        return sobol_noise


    def get_sample_joint(self, samples: torch.Tensor, q: torch.Tensor, qdot: torch.Tensor, dt):
        qdot0 = qdot.unsqueeze(0).unsqueeze(0).expand(self.n_sample, 1, self.n_action)  # (n_sample, 1, n_action)
        q0 = q.unsqueeze(0).unsqueeze(0).expand(self.n_sample, 1, self.n_action)        # (n_sample, 1, n_action)
        v = torch.cumsum(samples * dt, dim=1) + qdot0                                   # (n_sample, n_horizon, n_action)
        v_prev = torch.cat([qdot0, v[:, :-1, :]], dim=1)                                # (n_sample, n_horizon, n_action)

        dq = v_prev * dt + 0.5 * samples * dt**2
        q = torch.cumsum(dq, dim=1) + q0
        return q


    def get_prev_sample_joint(self, u_prev: torch.Tensor, q: torch.Tensor, qdot: torch.Tensor, dt):
        qdot0 = qdot.unsqueeze(0).expand(1, self.n_action)  # (1, n_action)
        q0 = q.unsqueeze(0).expand(1, self.n_action)        # (1, n_action)
        v = torch.cumsum(u_prev * dt, dim=0) + qdot0        # (n_horizon, n_action)
        v_prev = torch.cat([qdot0, v[:-1, :]], dim=0)       # (n_horizon, n_action)

        dq = v_prev * dt + 0.5 * u_prev * dt**2
        q = torch.cumsum(dq, dim=0) + q0
        return q.unsqueeze(0), v_prev



    def update_distribution(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, noise: torch.Tensor):
        if self.sigma_update:
            self.sigma, self.sigma_matrix = self.updater.update(self.sigma, self.init_sigma, self.kappa_eye, u, v, w, noise)
        return
    

    def n_sample_sampling(self, n_sample):
        n_total_sample = n_sample * self.n_horizon * self.n_action
        sobol_points = self.sobol_engine.draw(n_total_sample).to(**self.tensor_args)
        sobol_noise = sobol_points[:,0].reshape(n_sample, self.n_horizon, self.n_action)
        return sobol_noise
    

    def n_sample_horizon_sampling(self, n_sample, n_knot):
        n_total_sample = n_sample * n_knot * self.n_action
        sobol_points = self.sobol_engine.draw(n_total_sample).to(**self.tensor_args)
        sobol_noise = sobol_points[:,0].reshape(n_sample, n_knot, self.n_action)
        return sobol_noise