import torch

from mppi_solver.src.solver.cost.pose_cost import PoseCost
from mppi_solver.src.solver.cost.covar_cost import CovarCost
from mppi_solver.src.solver.cost.action_cost import ActionCost
from mppi_solver.src.solver.cost.joint_space_cost import JointSpaceCost
from mppi_solver.src.solver.cost.collision_cost import CollisionAvoidanceCost
from mppi_solver.src.solver.cost.stop_cost import StopCost
from mppi_solver.src.solver.cost.ee_cost import EECost
from mppi_solver.src.solver.cost.reference_cost import ReferenceCost
from mppi_solver.src.solver.cost.base_disturbance_cost import BaseDisturbanceCost

from mppi_solver.src.utils.pose import Pose

from rclpy.logging import get_logger

class CostManager:
    def __init__(self, params, tensor_args):
        self.logger = get_logger("Cost_Manager")

        # MPPI Parameter
        self.tensor_args = tensor_args
        self.n_sample = params['mppi']['sample']
        self.n_horizon = params['mppi']['horizon']
        self.n_action = params['mppi']['action']
        self._lambda = params['mppi']['_lambda']
        self.alpha = params['mppi']['alpha']
        self.gamma = params['mppi']['gamma']
        self.dt = params['mppi']['dt']

        ## Weights
        self.pose_cost_weights = params['cost']['pose']                     # Pose Cost Weights
        self.covar_weights = params['cost']['covariance']                   # Covariance Cost Weights
        self.action_weights = params['cost']['action']                      # Action Cost Weights
        self.joint_space_weights = params['cost']['joint_space']            # Joint Space Cost Weights
        self.collision_weights = params['cost']['collision']                # Collision Avoidance Cost Weights
        self.stop_weights = params['cost']['stop']                          # Stop Cost Weights
        self.ee_weights = params['cost']['end_effector']                    # End-effector Cost Weights
        self.reference_weights = params['cost']['reference']                # Reference Cost Weights
        self.base_disturbance_weights = params['cost']['base_disturbance']  # Reference Cost Weights

        # Cost Library
        self.pose_cost = PoseCost(self.pose_cost_weights, self.gamma, self.n_horizon, self.tensor_args)
        self.covar_cost = CovarCost(self.covar_weights, self._lambda, self.alpha, self.tensor_args)
        self.action_cost = ActionCost(self.action_weights, self.gamma, self.n_horizon, self.tensor_args)
        self.joint_cost = JointSpaceCost(self.joint_space_weights, self.gamma, self.n_horizon, self.tensor_args)
        self.collision_cost = CollisionAvoidanceCost(self.collision_weights, self.gamma, self.n_horizon, self.tensor_args)
        self.stop_cost = StopCost(self.stop_weights, self.gamma, self.n_horizon, self.dt, self.tensor_args)
        self.ee_cost = EECost(self.ee_weights, self.gamma, self.n_horizon, self.dt, self.tensor_args)
        self.reference_cost = ReferenceCost(self.reference_weights, self.gamma, self.n_horizon, self.dt, self.tensor_args)
        self.disturbace_cost = BaseDisturbanceCost(self.base_disturbance_weights, self.gamma, self.n_horizon, self.tensor_args)


        # For Pose Cost
        self.target : Pose
        self.eef_trajectories : torch.Tensor
        self.joint_trajectories : torch.Tensor
        self.pose_trajectories : torch.Tensor
        self.qSamples : torch.Tensor
        self.uSamples : torch.Tensor

        # For Covar Cost
        self.u : torch.Tensor
        self.v : torch.Tensor
        self.sigma_matrix : torch.Tensor

        # For Disturbance Cost
        self.base_pose : Pose

        # For Collision Cost
        self.collision_target : torch.Tensor

        # For End-effector Cost
        self.jacobian : torch.Tensor
        self.target_dist : torch.Tensor
        self.link_list : torch.Tensor

        # For Base Disturbance Cost
        self.jacob_bm : torch.Tensor


    def update_pose_cost(self, qSamples: torch.Tensor, uSamples: torch.Tensor, vSamples: torch.Tensor, 
                         eef_trajectories: torch.Tensor, joint_trajectories: torch.Tensor, pose_trajectories: torch.Tensor, target: Pose):
        self.target = target.clone()
        self.qSamples = qSamples.clone()
        self.uSamples = uSamples.clone()
        self.vSamples = vSamples.clone()
        self.eef_trajectories = eef_trajectories.clone()
        self.joint_trajectories = joint_trajectories.clone()
        self.pose_trajectories = pose_trajectories.clone()


    def update_covar_cost(self, u: torch.Tensor, v: torch.Tensor, sigma_matrix: torch.Tensor):
        self.u = u.clone()
        self.v = v.clone()
        self.sigma_matrix = sigma_matrix.clone()


    def update_base_cost(self, base_pose: Pose, q: torch.Tensor):
        self.base_pose = base_pose
        self.test_joint = q


    def update_collision_cost(self, collision_target: torch.Tensor):
        self.collision_target = collision_target.clone()


    def update_ee_cost(self, jacobian: torch.Tensor, target_dist: torch.Tensor):
        self.jacobian = jacobian.clone()
        self.target_dist = target_dist.clone()


    def update_reference_cost(self, link_list: torch.Tensor):
        self.link_list = link_list.clone()


    def update_base_disturbance_cost(self, jacobian_bm : torch.Tensor):
        self.jacob_bm = jacobian_bm.clone()

    
    def compute_all_cost(self):
        S = torch.zeros((self.n_sample), **self.tensor_args)

        S += self.pose_cost.compute_stage_cost(self.eef_trajectories, self.target)
        S += self.pose_cost.compute_terminal_cost(self.eef_trajectories, self.target)
        # S += self.covar_cost.compute_covar_cost(self.sigma_matrix, self.u, self.v)
        # S += self.joint_cost.compute_centering_cost(self.qSamples)
        # S += self.joint_cost.compute_jointTraj_cost(self.qSamples, self.joint_trajectories)
        S += self.action_cost.compute_action_cost(self.uSamples)
        S += self.collision_cost.compute_collision_cost(self.base_pose, self.qSamples, self.collision_target)
        S += self.stop_cost.compute_stop_cost(self.vSamples)
        S += self.ee_cost.compute_ee_cost(self.vSamples, self.jacobian, self.target_dist)
        S += self.reference_cost.compute_reference_cost(self.link_list, self.pose_trajectories)
        S += self.disturbace_cost.compute_base_disturbance_cost(self.jacob_bm, self.vSamples)
        return S
    