import os
import torch
from rclpy.logging import get_logger
from mppi_solver.src.utils.pose import Pose
from ament_index_python.packages import get_package_share_directory

from mppi_solver.src.solver.cost.pts.network import *


class CollisionAvoidanceCost():
    def __init__(self, params, gamma: float, n_horizon: int, tensor_args):
        self.logger = get_logger("Joint_Space_Cost")
        self.tensor_args = tensor_args
        self.device = self.tensor_args['device']
        self.n_horizon = n_horizon
        self.gamma = gamma

        self.collision_weight = params['weight']
        self.collision_softcap = params['softcap']

        # Neural network for calculate distances
        # self.model = MLPRegressionNormDropout().to(**self.tensor_args)
        # pt_path = os.path.join(get_package_share_directory("mppi_solver"), "pts", "best_canadarm_MLPRegressionDropout.pt")

        self.model = MLPWithResidualNormELU().to(**self.tensor_args)
        pt_path = os.path.join(get_package_share_directory("mppi_solver"), "pts", "best_canadarm_MLPWithResidualNorm.pt")

        checkpoint = torch.load(pt_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # variables for calculate cost
        self.mount_tf = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 3.6], [0.0, 0.0, 0.0, 1.0]], **self.tensor_args)
        self.zero = torch.tensor(0.0, **self.tensor_args)
        self.softcap = torch.tensor(self.collision_softcap, **self.tensor_args)
        return


    def compute_collision_cost(self, base_pose: Pose, qSamples: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if torch.allclose(targets, torch.tensor([[0.0, 0.0, 0.0]], **self.tensor_args)):
            return torch.zeros((qSamples.shape[0]), **self.tensor_args)
        base_joint_pose = base_pose.tf_matrix(self.tensor_args) @ self.mount_tf

        # calculate distance
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                n_sample, _ ,_ = qSamples.shape
                n_targets, _ = targets.shape

                ones = torch.ones(n_targets, 1, **self.tensor_args)
                targets_4 = torch.cat([targets, ones], dim=1)

                target_transform = (base_joint_pose @ targets_4.T).T
                w       = target_transform[:, 3:4]
                p_trans = target_transform[:, :3] / w

                input = torch.cat([
                    qSamples.clone().repeat_interleave(n_targets, dim=0),
                    p_trans.repeat(n_sample, 1).view(-1, 1, targets.size(-1)).expand(-1, self.n_horizon, targets.size(-1))
                ], dim=-1)

                output = 0.01 * self.model(input) # train NN by multiplying the training data by 100
                dist = output.unflatten(0, (n_sample, n_targets)).permute(0, 2, 1, 3).to(**self.tensor_args) # (n_sample, n_horizon, n_target, n_dof+1))

        cost_collision = torch.sum(torch.max(self.zero, -torch.log(dist) + self.softcap), dim=(2,3))
        cost_collision = self.collision_weight * cost_collision

        gamma = self.gamma ** torch.arange(self.n_horizon, **self.tensor_args)
        cost_collision = cost_collision * gamma

        cost_collision = torch.sum(cost_collision, dim=1)
        return cost_collision
    

    def compute_prev_collision_cost(self, base_pose: Pose, qSamples: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if torch.allclose(targets, torch.tensor([[0.0, 0.0, 0.0]], **self.tensor_args)):
            return torch.zeros((qSamples.shape[0]), **self.tensor_args)
        base_joint_pose = base_pose.tf_matrix(self.tensor_args) @ self.mount_tf

        # calculate distance
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                n_targets, _ = targets.shape
                
                ones = torch.ones(n_targets, 1, **self.tensor_args)
                targets_4 = torch.cat([targets, ones], dim=1)

                target_transform = (base_joint_pose @ targets_4.T).T
                w       = target_transform[:, 3:4]
                p_trans = target_transform[:, :3] / w

                input = torch.cat([
                    qSamples.clone().repeat_interleave(n_targets, dim=0),
                    p_trans.view(-1, 1, targets.size(-1)).expand(-1, self.n_horizon, targets.size(-1))
                ], dim=-1)

                output = 0.01 * self.model(input) # train NN by multiplying the training data by 100
                dist = output.permute(1, 0, 2).to(**self.tensor_args) # (n_horizon, n_target, n_dof+1))

        cost_collision = torch.sum(torch.max(self.zero, -torch.log(dist) + self.softcap), dim=(1, 2))
        cost_collision = self.collision_weight * cost_collision

        gamma = self.gamma ** torch.arange(self.n_horizon, **self.tensor_args)
        cost_collision = cost_collision * gamma

        cost_collision = torch.sum(cost_collision, dim=0)
        return cost_collision
    