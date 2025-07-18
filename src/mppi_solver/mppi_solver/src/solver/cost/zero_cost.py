import torch
from rclpy.logging import get_logger
from mppi_solver.src.utils.pose import Pose


class ZeroCost:
    def __init__(self, params, gamma, n_horizon, dt, tensor_args):
        self.logger = get_logger("Zero_Cost")
        self.tensor_args = tensor_args

        self.n_horizon = n_horizon
        self.gamma = gamma
        self.dt = dt

        self.zero_weight = params['weight']
        self.v_max = params['v_max']
        self.vmax = torch.tensor(self.v_max, **tensor_args).unsqueeze(0)
        self.softcap = params['softcap']
        self.softcap = torch.tensor(self.softcap, **tensor_args)


    def compute_zero_cost(self, uSample: torch.Tensor, v_prev: torch.Tensor, eefTraj: torch.Tensor, target_pose: Pose):
        # vel = v_prev[0].unsqueeze(0) + torch.cumsum(uSample * self.dt, dim=1)
        # abs_zero_vel = torch.where(torch.abs(vel) < self.vmax, 0.0, torch.abs(vel))

        # ee_sample_pose = eefTraj[:,:,0:3,3].clone()
        # target = target_pose.pose.to(**self.tensor_args)

        # goal_dist = torch.norm(eefTraj[:,:,0:3,3] - target, p=2, dim=2).unsqueeze(2).repeat(1, 1, uSample.shape[2]).to(**self.tensor_args) # torch.Size([1000, 31, 7])
        
        # self.logger.info(f"goal dist: {goal_dist.shape}")
        # self.logger.info(f"goal dist: {goal_dist}")

        # softcap = self.softcap.repeat(uSample.shape[0], uSample.shape[1], uSample.shape[2])

        # if self.zero_weight > 0.0:
        #     zero_cost = torch.where(goal_dist <= softcap, abs_zero_vel, 0.0)
        #     zero_cost = torch.norm(zero_cost, p=2, dim=2)
        #     zero_cost = self.zero_weight * zero_cost
        # else:
        #     zero_cost = torch.zeros((uSample.shape[0], uSample.shape[1]), **self.tensor_args)
        zero_cost = torch.zeros((uSample.shape[0], uSample.shape[1]), **self.tensor_args)

        gamma = self.gamma ** torch.arange(self.n_horizon, **self.tensor_args)
        zero_cost = zero_cost * gamma

        zero_cost = torch.sum(zero_cost, dim=1)
        return zero_cost
    

    def compute_prev_zero_cost(self, uSample: torch.Tensor, v_prev: torch.Tensor, eefTraj: torch.Tensor, target_pose: Pose):
        # vel = v_prev[0].unsqueeze(0) + torch.cumsum(uSample * self.dt, dim=0)
        # abs_zero_vel = torch.where(torch.abs(vel) < self.vmax, 0.0, torch.abs(vel))

        # ee_sample_pose = eefTraj[:,0:3,3].clone()
        # target = target_pose.pose.to(**self.tensor_args)

        # goal_dist = torch.norm(eefTraj[:,0:3,3], p=2, dim=1).unsqueeze(0).repeat(1, uSample.shape[1]).to(**self.tensor_args)
        # softcap = self.softcap.repeat(uSample.shape[0], uSample.shape[1])

        # if self.zero_weight > 0.0:
        #     zero_cost = torch.where(goal_dist <= self.softcap, abs_zero_vel, 0.0)
        #     zero_cost = torch.norm(zero_cost, p=2, dim=1)
        #     zero_cost = self.zero_weight * zero_cost
        # else:
        #     zero_cost = torch.zeros((uSample.shape[0]), **self.tensor_args)
        zero_cost = torch.zeros((uSample.shape[0]), **self.tensor_args)

        gamma = self.gamma ** torch.arange(self.n_horizon, **self.tensor_args)
        zero_cost = zero_cost * gamma

        zero_cost = torch.sum(zero_cost, dim=0)
        return zero_cost
    





# class ZeroCost(nn.Module):
#     def __init__(self, device=torch.device('cpu'), float_dtype=torch.float64,
#                  hinge_val=100.0, weight=1.0, gaussian_params={}, max_vel=0.01):
#         super(ZeroCost, self).__init__()
#         self.device = device
#         self.float_dtype = float_dtype
#         self.Z = torch.zeros(1, device=self.device, dtype=self.float_dtype)
#         self.weight = torch.as_tensor(weight, device=device, dtype=float_dtype)
#         self.proj_gaussian = GaussianProjection(gaussian_params=gaussian_params)
#         self.hinge_val = hinge_val
#         self.max_vel = max_vel


#     def forward(self, vels, goal_dist):
#         inp_device = vels.device
#         vel_err = torch.abs(vels.to(self.device))
#         goal_dist = goal_dist.to(self.device)
#         # max velocity threshold:
#         vel_err[vel_err < self.max_vel] = 0.0
#         if(self.hinge_val > 0.0):
#             vel_err = torch.where(goal_dist <= self.hinge_val, vel_err, 0.0 * vel_err / goal_dist) #soft hinge
#         cost = self.weight * self.proj_gaussian((torch.sum(torch.square(vel_err), dim=-1)))
#         return cost.to(inp_device)
