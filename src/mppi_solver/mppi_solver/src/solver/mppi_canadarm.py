# Python 
import os
import math
import yaml
import time
from datetime import datetime
from typing import Tuple

# Linear Algebra
import numpy as np
import torch
import torch.nn.functional as F

# RCLPY 
from rclpy.logging import get_logger
from ament_index_python.packages import get_package_share_directory

# Sampling Library
from mppi_solver.src.solver.sampling.mixed_noise import MixedSampling

# Cost Library
from mppi_solver.src.solver.cost.cost_manager import CostManager

# Kinematics Library
from mppi_solver.src.robot.urdfFks.urdf_forward_kinematics import URDFForwardKinematics
from mppi_solver.src.robot.ik.canadarm_jacob import CanadarmJacob

# TF Library
from mppi_solver.src.utils.pose import Pose, pose_diff, pos_diff
from mppi_solver.src.utils.rotation_conversions import euler_angles_to_matrix, matrix_to_euler_angles, quaternion_to_matrix, matrix_to_quaternion

# Filter : MPPI
from mppi_solver.src.solver.filter.svg_filter import SavGolFilter

# Logger : MATLAB for Plot
from pathlib import Path
from mppi_solver.src.utils.matlab_logger import MATLABLogger
from mppi_solver.src.utils.time import Time

# Chan 250804
from mppi_solver.src.solver.solver_v2.sampling.standard_normal_noise import StandardSamplling

class MPPI():
    def __init__(self, params):
        self.logger = get_logger("MPPI")

        # Load parameters
        self.params = params

        # torch env
        os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info('Device: ' + self.device.type)
        torch.set_default_dtype(torch.float64)
        self.tensor_args = {'device': self.device, 'dtype': torch.float64}
        self.is_compile = self.params['mppi']['compile']

        # Sampling parameters
        self.is_free_floating = params['mppi']['free_floating']
        self.is_base_move = params['mppi']['base_move']

        self.n_action = self.params['mppi']['action']
        self.n_manipulator_dof = self.params['mppi']['manipulator_dof']
        self.n_mobile_dof = self.params['mppi']['mobile_dof']
        self.n_samples = self.params['mppi']['sample']
        self.n_horizon = self.params['mppi']['horizon']
        self.dt = self.params['mppi']['dt']

        # Manipulator states
        self._q = torch.zeros(self.n_action, **self.tensor_args)
        self._qdot = torch.zeros(self.n_action, **self.tensor_args)
        self._qddot = torch.zeros(self.n_action, **self.tensor_args)

        self.q_prev = torch.zeros(self.n_action, **self.tensor_args)
        self.v_prev = torch.zeros(self.n_action, **self.tensor_args)

        self.ee_pose = Pose()
        self.eefTraj = torch.zeros((self.n_samples, self.n_horizon, 4, 4), **self.tensor_args)

        # Action
        self.u = torch.zeros((self.n_action), **self.tensor_args)
        self.u_prev = torch.zeros((self.n_horizon, self.n_action), **self.tensor_args)

        # Sampling class
        # self.sample_gen = MixedSampling(self.params, self.tensor_args)

        # base control states
        self.base_pose = Pose(self.tensor_args)

        # Target states
        self.target_pose = Pose(self.tensor_args)
        self.target_pose.pose = torch.tensor([0.5, 0.0, 0.6], **self.tensor_args)
        self.target_pose.orientation = torch.tensor([0.0, 0.0, 0.0, 1.0], **self.tensor_args)
        self.predict_target_pose = torch.zeros((self.n_horizon, 6))

        self.diff_ori_3d = torch.zeros((3), **self.tensor_args)
        self.target_dist = torch.zeros((1), **self.tensor_args)

        # MPPI Cost Manager
        self._lambda = params['mppi']['_lambda']
        self.cost_manager = CostManager(self.params, self.tensor_args)

        # Import URDF for forward kinematics
        package_name = params['mppi']['package_name']
        urdf_name = params['mppi']['urdf_name']
        urdf_file_path = os.path.join(get_package_share_directory(package_name), "models", "canadarm", urdf_name)

        self.fk_canadarm = URDFForwardKinematics(params=self.params, urdf=urdf_file_path, root_link='Base_SSRMS', end_links='EE_SSRMS_tip', tensor_args=self.tensor_args)

        # Filter
        self.svg_filter = SavGolFilter()

        # Align the target pose
        self.is_align = False

        # Collision avoidance target (debris)
        self.collision_target = torch.tensor([[0.0, 0.0, 0.0]], **self.tensor_args)

        # 250519
        self.noise = None
        self.noise_prev = torch.zeros((self.n_samples, self.n_horizon, self.n_action), **self.tensor_args)
        self.uSamples = None
        self.param_gamma = self._lambda * (1.0 - 0.9)
        self.is_reaching = False

        self.calc_jacob = CanadarmJacob(params, self.tensor_args)

        # Log
        self.sim_time = Time()
        self.matlab_logger = MATLABLogger(script_name=Path(__file__).stem, file_name="mppi_canadarm")
        self.matlab_logger.create_dataset(dataset_name="joint_angle", shape=8)
        self.matlab_logger.create_dataset(dataset_name="joint_angular_velocity", shape=8)
        self.matlab_logger.create_dataset(dataset_name="endeffector_pose", shape=7)
        self.matlab_logger.create_dataset(dataset_name="target_pose", shape=7)
        self.matlab_logger.create_dataset(dataset_name="torque", shape=8)
        self.matlab_logger.create_dataset(dataset_name="torque_clamp", shape=8)
        self.matlab_logger.create_dataset(dataset_name="torque_energy", shape=8)
        self.matlab_logger.create_dataset(dataset_name="kinetic_energy", shape=2)
        self.matlab_logger.create_dataset(dataset_name="base_motion", shape=7)

        # self.matlab_logger.create_dataset(dataset_name="cost", shape=10)
        # self.matlab_logger.create_dataset(dataset_name="sigma", shape=(self.n_action+1))
        # self.matlab_logger.create_dataset(dataset_name="base", shape=7)

        # test
        # from torch.utils.tensorboard import SummaryWriter
        # self.tensor_board_logdir = "/home/user/space_ws/src/mppi_solver/mppi_solver/runs/weights"
        # self.writer = SummaryWriter(log_dir=self.tensor_board_logdir)
        # self.step = 0

        self.reference_joint = None
        self.reference_se3 = None
        self.iteration = 0

        # Torque
        self.state_M = None
        self.state_G = None
        self.torque_sum = torch.zeros((self.n_action), **self.tensor_args)

        # Chan
        self.sampling = StandardSamplling(self.n_samples, self.n_horizon, self.n_action, self.device)
        

    def check_reach(self):
        self.fk_canadarm.set_samples_and_timesteps(n_samples=1, n_horizon=1)
        ee_pose_mat, _, _ = self.fk_canadarm(self._q[self.n_mobile_dof:], 
                'EE_SSRMS_tip', 'Base_SSRMS', init_transformation=self.base_pose.tf_matrix(self.tensor_args))
        self.fk_canadarm.set_samples_and_timesteps(n_samples=self.n_samples, n_horizon=self.n_horizon)
        
        self.ee_pose.from_matrix(ee_pose_mat.squeeze(0).squeeze(0))

        # pose_err = pos_diff(self.ee_pose, self.target_pose)
        ee_ori_mat = euler_angles_to_matrix(self.ee_pose.rpy, "ZYX")
        target_ori_mat = euler_angles_to_matrix(self.target_pose.rpy, "ZYX")
        diff_ori_mat = torch.matmul(torch.linalg.inv(ee_ori_mat), target_ori_mat)
        # diff_ori_quat = matrix_to_quaternion(diff_ori_mat)
        self.diff_ori_3d = matrix_to_euler_angles(diff_ori_mat, "ZYX")
        self.target_dist = torch.norm(self.ee_pose.pose - self.target_pose.pose, p=2)
        
        # self.logger.info(f"EE : {ee_pose_mat}")
        # self.logger.info(f"Target : {self.target_dist}")
        # self.logger.info(f"target dist: {self.target_dist}")
        # self.logger.info(f"target pose: {target_ori_mat}")
        # self.logger.info(f"EE POSE : {self.ee_pose.pose[0]:.2f}, {self.ee_pose.pose[1]:.2f}, {self.ee_pose.pose[2]:.2f}, {self.ee_pose.orientation[0]:.2f}, {self.ee_pose.orientation[1]:.2f}, {self.ee_pose.orientation[2]:.2f}, {self.ee_pose.orientation[3]:.2f}")

        # Chan
        self.cost_manager.update_weights(self.target_dist)
        # self.logger.info(f"Target Dist : {self.target_dist}")

        if self.target_dist < 0.01:
            return True
        else:
            return False
        

    def compute_control_input(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.check_reach():
            return self.qdes, self.vdes, self.u
        
        self.MATLAB_log()

        u = self.u_prev.clone()
        # noise = self.sample_gen.sampling()
        noise = self.sampling.sampling()
        v = u + noise

        qSamples, vSamples = self.sampling.get_sample_joint(v, self._q, self._qdot, self.dt)
        qSamples = qSamples.to(**self.tensor_args)
        vSamples = vSamples.to(**self.tensor_args)

        trajectory, link_list, com_list = self.fk_canadarm(qSamples,
                                        'EE_SSRMS_tip', 'Base_SSRMS', self.base_pose.tf_matrix(self.tensor_args))

        jacob = self.calc_jacob(com_list, link_list, bm=False)
        jacob_bm, H_star = self.calc_jacob(com_list, link_list, jacob, bm=True)

        torque = torch.einsum('ij,btj->bti', self.state_M, v) + self.state_G

        # self.logger.info(f"Traj : {trajectory}")
        self.cost_manager.update_pose_cost(qSamples, v, vSamples, trajectory, self.reference_joint, self.reference_se3, self.target_pose)
        self.cost_manager.update_covar_cost(u, v, self.sampling.sigma_matrix)
        self.cost_manager.update_collision_cost(self.base_pose, self.collision_target)
        # self.cost_manager.update_ee_cost(jacob, self.target_dist)
        # self.cost_manager.update_reference_cost(link_list[...,-2])
        self.cost_manager.update_base_disturbance_cost(jacob_bm)
        self.cost_manager.update_energy_cost(torque, H_star)

        S = self.cost_manager.compute_all_cost()

        w = self.compute_weights(S, self._lambda)
        w_expanded = w.view(-1, 1, 1)
        w_eps = torch.sum(w_expanded * noise, dim = 0)
        # w_eps = self.svg_filter.savgol_filter_torch(w_eps, window_size=9, polyorder=2, tensor_args=self.tensor_args)

        u += w_eps

        # self.sample_gen.update_distribution(u, v, w, noise)

        self.u_prev = u.clone()
        self.u = u[0].clone()
        self.noise_prev = noise.clone()

        self.vdes = self._qdot + self.u * self.dt
        self.qdes = self._q + self._qdot * self.dt + 0.5 * self.u * self.dt * self.dt
        return self.qdes, self.vdes, self.u


    def compute_weights(self, S: torch.Tensor, _lambda: float) -> torch.Tensor:
        weights = torch.softmax(-S / _lambda, dim=0)  # (n_samples,)
        return weights
    

    def set_joint(self, joint_states: torch.Tensor):
        joint_states = joint_states.to(**self.tensor_args)
    
        self._q = joint_states[:, 0]
        self._qdot = joint_states[:, 1]
        self._qddot = joint_states[:, 2]
        return

    def set_target_pose(self, pose: Pose):
        self.target_pose.pose = pose.pose.to(**self.tensor_args)
        self.target_pose.orientation = pose.orientation.to(**self.tensor_args)

        # self.logger.info(f"Target Ori : {self.target_pose.orientation}")
        return
    
    def set_base_pose(self, pos, ori):
        self.base_pose.pose = pos
        self.base_pose.orientation = ori
        self.calc_jacob.base_update(self.base_pose.tf_matrix())
        # self.logger.info(f"Base Pose Mat : {self.base_pose.tf_matrix()}")
        return
    
    def set_collision_target(self, target: torch.Tensor):
        self.collision_target = target.clone().to(**self.tensor_args)
        return
    
    def setReference(self, reference_joint: torch.Tensor, reference_se3: torch.Tensor):
        self.reference_joint = reference_joint.clone()
        self.reference_se3 = reference_se3.clone()
        return

    def setstate_mass_nle(self, M: np.ndarray, G: np.ndarray):
        self.state_M = torch.from_numpy(M).to(**self.tensor_args)
        self.state_G = torch.from_numpy(G).to(**self.tensor_args)

    def warm_up(self):
        qSamples = torch.zeros((1, self.n_horizon, self.n_action), **self.tensor_args)
        self.fk_canadarm.set_samples_and_timesteps(n_samples=1, n_horizon=self.n_horizon)
        self.fk_canadarm(qSamples, 'EE_SSRMS_tip', 'Base_SSRMS', self.base_pose.tf_matrix(self.tensor_args))
        self.fk_canadarm.set_samples_and_timesteps(n_samples=self.n_samples, n_horizon=self.n_horizon)

        qSamples = torch.zeros((self.n_samples, self.n_horizon, self.n_action), **self.tensor_args)
        _, link_list, com_list = self.fk_canadarm(qSamples, 'EE_SSRMS_tip', 'Base_SSRMS', self.base_pose.tf_matrix(self.tensor_args))

        jaco = self.calc_jacob(com_list, link_list, bm=False)
        self.calc_jacob(com_list, link_list, jaco, bm=True)
        return


    def MATLAB_log(self):
        q_prev, v_prev = self.sampling.get_prev_sample_joint(self.u_prev, self._q, self._qdot, self.dt)
        self.fk_canadarm.set_samples_and_timesteps(n_samples=1, n_horizon=self.n_horizon)
        ee_traj_prev, link_list_prev, com_list_prev = self.fk_canadarm(q_prev,
                                                    'EE_SSRMS_tip', 'Base_SSRMS', self.base_pose.tf_matrix(self.tensor_args))
        self.fk_canadarm.set_samples_and_timesteps(n_samples=self.n_samples, n_horizon=self.n_horizon)
        ee_jacobian_prev = self.calc_jacob(com_list_prev, link_list_prev, bm=False)
        jacob_bm, H_star = self.calc_jacob(com_list_prev, link_list_prev, ee_jacobian_prev, bm=True)

        v_prev = v_prev.squeeze(0)
        H_star = H_star.squeeze(0)

        # Torque
        torque = self.state_M @ self.u_prev[0] + self.state_G
        torque_clamp = torch.clamp(torque, -2332, 2332)
        torque_energy = torch.einsum('a,ha->ha', torque_clamp, v_prev)
        kinetic_energy = 0.5 * torch.einsum('hi,hij,hj->h', v_prev, H_star, v_prev)

        self.matlab_logger.log("joint_angle", [self.sim_time.time] + self._q.tolist())
        self.matlab_logger.log("joint_angular_velocity", [self.sim_time.time] + self._qdot.tolist())
        self.matlab_logger.log("endeffector_pose", [self.sim_time.time] + self.ee_pose.np_pose.tolist() + self.ee_pose.np_rpy.tolist())
        self.matlab_logger.log("target_pose", [self.sim_time.time] + self.target_pose.np_pose.tolist() + self.target_pose.np_rpy.tolist())
        self.matlab_logger.log("torque", [self.sim_time.time] + torque.tolist())
        self.matlab_logger.log("torque_clamp", [self.sim_time.time] + torque_clamp.tolist())
        self.matlab_logger.log("torque_energy", [self.sim_time.time] + torque_energy[0,:].tolist())
        self.matlab_logger.log("kinetic_energy", [self.sim_time.time, kinetic_energy[0].tolist()])
        self.matlab_logger.log("base_motion", [self.sim_time.time] + self.base_pose.np_pose.tolist() + self.base_pose.np_rpy.tolist())
        return


    # def MATLAB_log(self):
    #     # q_prev, self.v_prev = self.sample_gen.get_prev_sample_joint(self.u_prev, self._q, self._qdot, self.dt)
    #     q_prev, self.v_prev = self.sampling.get_prev_sample_joint(self.u_prev, self._q, self._qdot, self.dt)

    #     self.fk_canadarm.set_samples_and_timesteps(n_samples=1, n_horizon=self.n_horizon)
    #     ee_traj_prev, link_list_prev, com_list_prev = self.fk_canadarm(q_prev,
    #                                                 'EE_SSRMS_tip', 'Base_SSRMS', self.base_pose.tf_matrix(self.tensor_args))
    #     self.fk_canadarm.set_samples_and_timesteps(n_samples=self.n_samples, n_horizon=self.n_horizon)
        
    #     ee_traj_prev = ee_traj_prev.squeeze(0).cpu()
    #     ee_jacobian_prev = self.calc_jacob(com_list_prev, link_list_prev, bm=False)
    #     jacob_bm, H_star = self.calc_jacob(com_list_prev, link_list_prev, ee_jacobian_prev, bm=True)
    #     ee_jacobian_prev = ee_jacobian_prev.squeeze(0)
    #     torque_prev = self.state_M @ self.u_prev[0] + self.state_G
    #     # self.torque_sum = self.torque_sum + torque_prev

    #     prev_stage_cost     = self.cost_manager.pose_cost.compute_prev_stage_cost(ee_traj_prev, self.target_pose)
    #     prev_terminal_cost  = self.cost_manager.pose_cost.compute_prev_terminal_cost(ee_traj_prev, self.target_pose)
    #     # prev_covar_cost     = self.cost_manager.covar_cost.compute_prev_covar_cost(self.sample_gen.sigma_matrix, self.u_prev, self.noise_prev)
    #     # prev_centering_cost = self.cost_manager.joint_cost.compute_prev_centering_cost(q_prev)
    #     # prev_tracking_cost  = self.cost_manager.joint_cost.compute_prev_jointTraj_cost(q_prev, self.reference_joint)
    #     prev_action_cost    = self.cost_manager.action_cost.compute_prev_action_cost(self.u_prev)
    #     prev_collision_cost = self.cost_manager.collision_cost.compute_prev_collision_cost(self.base_pose, q_prev, self.collision_target)
    #     prev_stop_cost      = self.cost_manager.stop_cost.compute_prev_stop_cost(self.v_prev)
    #     prev_ee_cost        = self.cost_manager.ee_cost.compute_prev_ee_cost(self.v_prev, ee_jacobian_prev, self.target_dist)
    #     prev_reference_cost = self.cost_manager.reference_cost.compute_prev_reference_cost(link_list_prev[...,-2], self.reference_se3)
    #     prev_disturbance_cost = self.cost_manager.disturbace_cost.compute_prev_base_disturbance_cost(jacob_bm, self.v_prev)
    #     prev_energy_cost    = self.cost_manager.energy_cost.compute_prev_energy_cost(torque_prev, self.v_prev, H_star)

    #     mean_prev_stage_cost     = torch.mean(prev_stage_cost)
    #     mean_prev_terminal_cost  = torch.mean(prev_terminal_cost)
    #     # mean_prev_covar_cost     = torch.mean(prev_covar_cost)
    #     # mean_prev_centering_cost = torch.mean(prev_centering_cost)
    #     # mean_prev_tracking_cost  = torch.mean(prev_tracking_cost)
    #     mean_prev_action_cost    = torch.mean(prev_action_cost)
    #     mean_prev_collision_cost = torch.mean(prev_collision_cost)
    #     mean_prev_stop_cost      = torch.mean(prev_stop_cost)
    #     mean_prev_ee_cost        = torch.mean(prev_ee_cost)
    #     mean_prev_reference_cost = torch.mean(prev_reference_cost)
    #     mean_prev_disturbance_cost = torch.mean(prev_disturbance_cost)
    #     mean_prev_energy_cost    = torch.mean(prev_energy_cost)
        
    #     self.matlab_logger.log("end_effector_pose", [self.sim_time.time] + self.ee_pose.np_pose.tolist() + self.ee_pose.np_rpy.tolist())
    #     self.matlab_logger.log("pos_err", [self.sim_time.time] + (self.ee_pose.np_pose - self.target_pose.np_pose).tolist())
    #     self.matlab_logger.log("ori_err", [self.sim_time.time] + self.diff_ori_3d.tolist())
    #     self.matlab_logger.log("cost", [self.sim_time.time] + [mean_prev_stage_cost.item(),
    #                                                            mean_prev_terminal_cost.item(),
    #                                                            mean_prev_action_cost.item(),
    #                                                            mean_prev_collision_cost.item(),
    #                                                            mean_prev_stop_cost.item(),
    #                                                            mean_prev_ee_cost.item(),
    #                                                            mean_prev_reference_cost.item(),
    #                                                            mean_prev_disturbance_cost.item(),
    #                                                            mean_prev_energy_cost.item()])
    #     self.matlab_logger.log("sigma", [self.sim_time.time] + torch.diag(self.sample_gen.sigma).tolist())
    #     self.matlab_logger.log("base", [self.sim_time.time] + \
    #                            self.cost_manager.disturbace_cost.compute_base_disturbance(jacob_bm, self.v_prev).tolist())
    #     self.matlab_logger.log("torque", [self.sim_time.time] + torque_prev.tolist())
    #     return