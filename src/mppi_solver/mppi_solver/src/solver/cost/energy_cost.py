import torch
from rclpy.logging import get_logger

"""
Combines key results from:
    • Wilde, Kwok Choon et al. (2018) “Equations of Motion of Free-Floating Spacecraft-Manipulator Systems: An Engineer's Tutorial”
        - Lagrangian derivation of total kinetic energy T (Eq. 27)
        - Conservation of base momentum ⇒ ẋ₀ = -H₀⁻¹ H₀m q̇  (Eqs. 42, 46)
        - Definition of generalized Jacobian J_bm ≔ -H₀⁻¹ H₀m
        - Reduced inertia matrix H* = H_m - H₀mᵀ H₀⁻¹ H₀m (Eq. 45)
        - Final joint-only energy: T = ½ q̇ᵀ H* q̇

    • Baressi Šegota, And̄elić et al. (2020) “Path planning optimization of six-degree-of-freedom robotic manipulators using evolutionary algorithms”
        - Parameterization of each joint trajectory as a 4th-order polynomial
        - Sampling M time points; computing joint torques tᵢₘ via Lagrange-Euler/Newton-Euler
        - Fitness function f(g) = ∑ₘ √(∑ᵢ tᵢₘ²) to minimize cumulative torque “energy”
        - Evolutionary optimizers: GA (avg./rand. crossover, 80% crossover rate, 1% mutation), SA (linear/geometric cooling), DE

Combined optimal-control cost:
    J_total = w₁ · (ŝᵀ·Dq) 
        + w₂ · ∫ₜ₀ᵗᶠ q̇ᵀ · J_bmᵀ·H_b·J_bm · q̇  dt

    where
        • ŝᵀ·Dq  is the torque-based energy term (Wilde §4.2, §4.3 ↔ Xie et al. Eqs. 25-26, 55, 59),
        • ∫ q̇ᵀ J_bmᵀ H_b J_bm q̇ dt  is the base-kinetic energy term (Wilde Eq. 27 & 45),
        • H_b and J_bm come from the free-floating spacecraft-manipulator dynamics tutorial,
        • Dq = [½ Δt² q̈ + Δt q̇]⃗ and ŝ = [s₁…sₘ]ᵀ from the torque-energy formulation,
        • w₁, w₂  are user-tunable weighting factors balancing torque vs. base-motion energy.
"""

class EnergyCost:
    def __init__(self, params, gamma, n_horizon, tensor_args):
        self.logger = get_logger("Energy_Cost")
        self.tensor_args = tensor_args
        self.n_horizon = n_horizon
        self.gamma = gamma
        self.gamma_horizon_gpu = self.gamma ** torch.arange(self.n_horizon, **self.tensor_args)

        self.torque_weight = params['torque_weight']
        self.kinetic_energy_weight = params['kinetic_energy_weight']


    def compute_energy_cost(self, torque: torch.Tensor, vSample: torch.Tensor, H_star: torch.Tensor):
        cost_torque = torch.einsum('sha,sha->sha', torque, vSample)
        cost_torque = torch.norm(cost_torque, p=2, dim=-1)

        cost_torque = self.torque_weight * cost_torque
        cost_torque = cost_torque * self.gamma_horizon_gpu

        cost_kinetic = 0.5 * torch.einsum('shi,shij,shj->sh', vSample, H_star, vSample)
        cost_kinetic = self.kinetic_energy_weight * cost_kinetic
        cost_kinetic = cost_kinetic * self.gamma_horizon_gpu

        cost_energy = cost_torque + cost_kinetic

        cost_energy = torch.sum(cost_energy, dim=1)
        return cost_energy
    

    def compute_prev_energy_cost(self, torque: torch.Tensor, vSample: torch.Tensor, H_star: torch.Tensor):
        cost_torque = torch.einsum('a,ha->ha', torque, vSample)
        cost_torque = torch.norm(cost_torque, p=2, dim=-1)

        cost_torque = self.torque_weight * cost_torque
        cost_torque = cost_torque * self.gamma_horizon_gpu

        cost_kinetic = 0.5 * torch.einsum('hi,hij,hj->h', vSample, H_star.squeeze(0), vSample)
        cost_kinetic = self.kinetic_energy_weight * cost_kinetic
        cost_kinetic = cost_kinetic * self.gamma_horizon_gpu

        cost_energy = cost_torque + cost_kinetic
        cost_energy = torch.sum(cost_energy, dim=0)
        return cost_energy
    