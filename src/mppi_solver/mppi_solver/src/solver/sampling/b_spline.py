import time
import torch
from rclpy.logging import get_logger

from mppi_solver.src.solver.sampling.distribution_updaters import StandardUpdater, CMAESUpdater


class BsplineSampling:
    def __init__(self, params, tensor_args):
        self.logger = get_logger("BSpline_Sampling")

        # Torch GPU
        self.tensor_args = tensor_args

        # Sampling Parameter
        self.n_sample  = params['mppi']['sample']
        self.n_horizon = params['mppi']['horizon']
        self.n_action  = params['mppi']['action']

        # knot
        self.knot_divider = params['sample']['bspline']['knot']
        self.n_knot = self.n_horizon // self.knot_divider
        self.n_total_sample = self.n_sample * self.n_knot * self.n_action

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


    def bspline_batch(self, ctrl_pts: torch.Tensor, n_knot: int, degree: int=2) -> torch.Tensor:
        knots = torch.cat([
                    torch.zeros(degree, **self.tensor_args),
                    torch.linspace(0.0, 1.0, steps=n_knot - degree + 1, **self.tensor_args),
                    torch.ones(degree, **self.tensor_args),
                ])
        u = torch.linspace(0.0, 1.0, steps=self.n_horizon, **self.tensor_args)

        N = torch.zeros((n_knot, self.n_horizon), **self.tensor_args)
        for i in range(n_knot):
            N[i] = ((u >= knots[i]) & (u < knots[i+1])).to(**self.tensor_args)

        for p in range(1, degree + 1):
            N_prev = N.clone()
            for i in range(self.n_knot - p):
                denom1 = knots[i+p] - knots[i]
                denom2 = knots[i+p+1] - knots[i+1]

                term1 = ((u - knots[i]) / denom1).clamp(min=0.0) * N_prev[i] if denom1 > 0 else 0.0
                term2 = ((knots[i+p+1] - u) / denom2).clamp(min=0.0) * N_prev[i+1] if denom2 > 0 else 0.0

                N[i] = term1 + term2

        output = torch.einsum('ska,kh->sah', ctrl_pts, N).permute(0, 2, 1)
        return output
    

    def sampling(self):
        standard_normal_noise = torch.randn(self.n_sample, self.n_knot, self.n_action, **self.tensor_args)
        bspline_noise = self.bspline_batch(standard_normal_noise, self.n_knot)

        self.sigma_matrix = self.sigma.expand(self.n_sample, self.n_horizon, -1, -1)
        noise = torch.matmul(bspline_noise.unsqueeze(-2), self.sigma_matrix).squeeze(-2)
        return noise
    

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
    

    def bspline_sampling(self, batch_sample: torch.Tensor, n_knot: int,n_sample: int):
        bspline_noise = self.bspline_batch(batch_sample, n_knot)

        self.sigma_matrix = self.sigma.expand(n_sample, self.n_horizon, -1, -1)
        noise = torch.matmul(bspline_noise.unsqueeze(-2), self.sigma_matrix).squeeze(-2)
        return noise
