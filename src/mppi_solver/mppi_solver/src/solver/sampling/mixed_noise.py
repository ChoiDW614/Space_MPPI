import math
import torch
from rclpy.logging import get_logger

from mppi_solver.src.solver.sampling.distribution_updaters import StandardUpdater, CMAESUpdater
from mppi_solver.src.solver.sampling.standard_normal_noise import StandardSampling
from mppi_solver.src.solver.sampling.halton_sequence_noise import HaltonSampling
from mppi_solver.src.solver.sampling.sobol_sequence_noise import SobolSampling
from mppi_solver.src.solver.sampling.b_spline import BsplineSampling

class MixedSampling:
    def __init__(self, params, tensor_args):
        self.logger = get_logger("Mixed_Sampling")

        # Torch GPU
        self.params = params
        self.tensor_args = tensor_args

        # Sampling Methods
        self.standard_normal = StandardSampling(self.params, self.tensor_args)
        self.halton_seq = HaltonSampling(self.params, self.tensor_args)
        self.sobol_seq = SobolSampling(self.params, self.tensor_args)
        self.bspline = BsplineSampling(self.params, self.tensor_args)

        # Sampling Parameter
        self.n_sample  = params['mppi']['sample']
        self.n_horizon = params['mppi']['horizon']
        self.n_action  = params['mppi']['action']

        self.sigma_scale: float = params['sample']['sigma_scale']
        self.sigma: torch.Tensor = torch.eye((self.n_action), **self.tensor_args) * self.sigma_scale
        self.init_sigma: torch.Tensor = self.sigma.clone()
        self.sigma_matrix = self.sigma.expand(self.n_sample, self.n_horizon, -1, -1)

        self.knot_divider = params['sample']['bspline']['knot']
        self.n_knot = self.n_horizon // self.knot_divider

        self.selection = params['sample']['selection']
        self.sel_weights = params['sample']['selection_weights']
        raw_counts = [w * self.n_sample for w in self.sel_weights]
        floor_counts = [math.floor(rc) for rc in raw_counts]

        remainder = self.n_sample - sum(floor_counts)
        fracs     = [rc - fc for rc, fc in zip(raw_counts, floor_counts)]
        idxs      = sorted(range(len(fracs)), key=lambda i: fracs[i], reverse=True)
        splits = floor_counts.copy()
        for i in idxs[:remainder]:
            splits[i] += 1
        self.splits = splits
        self.std_ref = self.standard_normal.n_sample_sampling(self.splits[0])

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
    

    def sampling(self, q: torch.Tensor=None):
        noise_list = []

        for sel, n_split in zip(self.selection, self.splits):
            # standard normal
            if sel == 'std':
                std_noise = self.standard_normal.n_sample_sampling(n_split)
                std_noise = self.scaling_noise(std_noise, n_split)
                scale, offset = self.compute_iqr_scale_and_offset(std_noise, ref_dist=self.std_ref)
                noise_list.append(std_noise * scale + offset)

            # Sobol sequence
            elif sel == 'sobol':
                sobol_noise = self.sobol_seq.n_sample_sampling(n_split)
                sobol_noise = self.scaling_noise(sobol_noise, n_split)
                scale, offset = self.compute_iqr_scale_and_offset(sobol_noise, ref_dist=self.std_ref)
                noise_list.append(sobol_noise * scale + offset)

            # standard Bspline
            elif sel == 'std_bspline':
                std_spline = self.standard_normal.n_sample_horizon_sampling(n_split, self.n_knot)
                std_bspline_noise = self.bspline.bspline_sampling(std_spline, self.n_knot, n_split)
                std_bspline_noise = self.scaling_noise(std_bspline_noise, n_split)
                scale, offset = self.compute_iqr_scale_and_offset(std_bspline_noise, ref_dist=self.std_ref)
                noise_list.append(std_bspline_noise * scale + offset)

            # Sobol Bspline
            elif sel == 'sobol_bspline':
                sobol_spline = self.sobol_seq.n_sample_horizon_sampling(n_split, self.n_knot)
                sobol_bspline_noise = self.bspline.bspline_sampling(sobol_spline, self.n_knot, n_split)
                sobol_bspline_noise = self.scaling_noise(sobol_bspline_noise, n_split)
                scale, offset = self.compute_iqr_scale_and_offset(sobol_bspline_noise, ref_dist=self.std_ref)
                noise_list.append(sobol_bspline_noise * scale + offset)

            # standard normal scaling
            elif sel == 'std_scaling':
                base = n_split // 3
                rem  = n_split - base * 3
                splits = [base + (1 if i < rem else 0) for i in range(3)]
                for r, ns in zip((0.75, 0.5, 0.25), splits):
                    std_noise = self.standard_normal.n_sample_ratio_sampling(ns, r)
                    std_noise = self.scaling_noise(std_noise, ns)
                    scale, offset = self.compute_iqr_scale_and_offset(std_noise, ref_dist=self.std_ref)
                    noise_list.append(std_noise + offset)

            elif sel == 'std_constacc_scaling' and q is not None:
                base = n_split // 3
                rem  = n_split - base * 3
                splits = [base + (1 if i < rem else 0) for i in range(3)]
                for r, ns in zip((0.75, 0.5, 0.25), splits):
                    cv = self.standard_normal.n_sample_horizon_sampling(ns, 1).expand(-1, self.n_horizon, -1) * r
                    cv_noise = self.scaling_noise(cv, ns)
                    scale, offset = self.compute_iqr_scale_and_offset(cv_noise, ref_dist=self.std_ref)
                    noise_list.append(cv_noise + offset)

            # std and Sobol constanct accel
            elif sel in ('std_constacc', 'sobol_constacc') and q is not None:
                if sel == 'std_constacc':
                    cv = self.standard_normal.n_sample_horizon_sampling(n_split, 1)
                else:
                    cv = self.sobol_seq.n_sample_horizon_sampling(n_split, 1)
                cv_full = cv.expand(-1, self.n_horizon, -1)
                cv_noise = self.scaling_noise(cv_full, n_split)
                
                scale, offset = self.compute_iqr_scale_and_offset(cv_noise, ref_dist=self.std_ref)
                noise_list.append(cv_noise * scale + offset)

        self.sigma_matrix = self.sigma.expand(self.n_sample, self.n_horizon, -1, -1) # For covar cost
        noise = torch.cat(noise_list, dim=0)
        return noise


    def get_sample_joint(self, samples: torch.Tensor, q: torch.Tensor, qdot: torch.Tensor, dt):
        qdot0 = qdot.unsqueeze(0).unsqueeze(0).expand(self.n_sample, 1, self.n_action)  # (n_sample, 1, n_action)
        q0 = q.unsqueeze(0).unsqueeze(0).expand(self.n_sample, 1, self.n_action)        # (n_sample, 1, n_action)
        v = torch.cumsum(samples * dt, dim=1) + qdot0                                   # (n_sample, n_horizon, n_action)
        v_prev = torch.cat([qdot0, v[:, :-1, :]], dim=1)                                # (n_sample, n_horizon, n_action)

        dq = v_prev * dt + 0.5 * samples * dt**2
        q = torch.cumsum(dq, dim=1) + q0
        return q, v


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
    

    def scaling_noise(self, noise: torch.Tensor, n_split: int):
        sigma_matrix = self.sigma.expand(n_split, self.n_horizon, -1, -1)
        noise = torch.matmul(noise.unsqueeze(-2), sigma_matrix).squeeze(-2)
        return noise


    def compute_iqr_scale_and_offset(self, samples: torch.Tensor, q_low: float = 0.25, q_high: float = 0.75,
                                    ref_q_low: float = 0.25, ref_q_high: float = 0.75, ref_dist: torch.Tensor = None
                                    ) -> tuple[float, float]:
        flat = samples.flatten()

        ql_src = torch.quantile(flat, q_low)
        qh_src = torch.quantile(flat, q_high)
        iqr_src = qh_src - ql_src

        if ref_dist is not None:
            flat_ref = ref_dist.flatten()
            ql_ref = torch.quantile(flat_ref, ref_q_low)
            qh_ref = torch.quantile(flat_ref, ref_q_high)
        else:
            ql_ref = ref_q_low
            qh_ref = ref_q_high

        iqr_ref = qh_ref - ql_ref

        scale = (iqr_ref / iqr_src).item()        # scale = IQR_ref / IQR_src
        offset = (ql_ref - ql_src * scale).item()  # offset = Q1_ref - Q1_src * scale
        return scale, offset
    
