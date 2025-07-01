import torch

# 수정필요
from mppi_solver.src.solver.cost.pose_cost import PoseCost
from mppi_solver.src.solver.cost.covar_cost import CovarCost
from mppi_solver.src.solver.cost.action_cost import ActionCost
from mppi_solver.src.solver.cost.joint_space_cost import JointSpaceCost



from mppi_solver.src.utils.pose import Pose

from rclpy.logging import get_logger

class CostManager:
    def __init__(self, params, device):
        self.logger = get_logger("Cost_Manager")

        # MPPI Parameter
        self.device = device
        self.n_sample = params['mppi']['sample']
        self.n_horizon = params['mppi']['horizon']
        self.n_action = params['mppi']['action']
        self._lambda = params['mppi']['_lambda']
        self.alpha = params['mppi']['alpha']
        self.gamma = params['mppi']['gamma']

        ## Weights
        self.pose_cost_weights = params['cost']['pose']             # Pose Cost Weights
        self.covar_weights = params['cost']['covariance']           # Covariance Cost Weights
        self.action_weights = params['cost']['action']              # Action Cost Weights
        self.joint_space_weights = params['cost']['joint_space']    # Joint Space Cost Weights

        # Cost Library
        self.pose_cost = PoseCost(self.pose_cost_weights, self.gamma, self.n_horizon, self.device)
        self.covar_cost = CovarCost(self.covar_weights, self._lambda, self.alpha, self.device)
        self.action_cost = ActionCost(self.action_weights, self.gamma, self.n_horizon, self.device)
        self.joint_cost = JointSpaceCost(self.joint_space_weights, self.gamma, self.n_horizon, self.device)

        # For Pose Cost
        self.target : Pose
        self.eef_trajectories : torch.Tensor
        self.joint_trajectories : torch.Tensor
        self.qSamples : torch.Tensor
        self.uSamples : torch.Tensor

        # For Covar Cost
        self.u : torch.Tensor
        self.v : torch.Tensor
        self.sigma_matrix : torch.Tensor


    def update_pose_cost(self, qSamples: torch.Tensor, uSamples: torch.Tensor, eef_trajectories: torch.Tensor, joint_trajectories: torch.Tensor,target: Pose):
        self.target = target.clone()
        self.qSamples = qSamples.clone()
        self.uSamples = uSamples.clone()
        self.eef_trajectories = eef_trajectories.clone()
        self.joint_trajectories = joint_trajectories.clone()


    def update_covar_cost(self,  u : torch.Tensor , v : torch.Tensor , sigma_matrix : torch.Tensor):
        self.u = u.clone()
        self.v = v.clone()
        self.sigma_matrix = sigma_matrix.clone()


    def compute_all_cost(self):
        S = torch.zeros((self.n_sample), device = self.device)

        S += self.pose_cost.compute_stage_cost(self.eef_trajectories, self.target)
        S += self.pose_cost.compute_terminal_cost(self.eef_trajectories, self.target)
        # S += self.covar_cost.compute_covar_cost(self.sigma_matrix, self.u, self.v)
        # S += self.joint_cost.compute_centering_cost(self.qSamples)
        S += self.joint_cost.compute_jointTraj_cost(self.qSamples, self.joint_trajectories)
        S += self.action_cost.compute_action_cost(self.uSamples)

        return S
    