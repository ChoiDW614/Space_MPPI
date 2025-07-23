# Python 
import os
import math
import yaml
import time
from datetime import datetime

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
from mppi_solver.src.robot.urdfFks.urdfFk import URDFForwardKinematics
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

        # Sampling parameters
        self.is_free_floating = params['mppi']['free_floating']
        self.is_base_move = params['mppi']['base_move']

        self.n_action = self.params['mppi']['action']
        self.n_manipulator_dof = self.params['mppi']['manipulator_dof']
        self.n_mobile_dof = self.params['mppi']['mobile_dof']
        self.n_samples = self.params['mppi']['sample']
        self.n_horizen = self.params['mppi']['horizon']
        self.dt = self.params['mppi']['dt']

        # Manipulator states
        self._q = torch.zeros(self.n_action, **self.tensor_args)
        self._qdot = torch.zeros(self.n_action, **self.tensor_args)
        self._qddot = torch.zeros(self.n_action, **self.tensor_args)

        self.q_prev = torch.zeros(self.n_action, **self.tensor_args)
        self.v_prev = torch.zeros(self.n_action, **self.tensor_args)

        self.ee_pose = Pose()
        self.eefTraj = torch.zeros((self.n_samples, self.n_horizen, 4, 4), **self.tensor_args)

        # Action
        self.u = torch.zeros((self.n_action), **self.tensor_args)
        self.u_prev = torch.zeros((self.n_horizen, self.n_action), **self.tensor_args)

        # Buffer
        self.buffer_size = 10
        self.weight_buffer = torch.zeros((self.buffer_size, self.n_samples), **self.tensor_args)
        self.action_buffer = torch.zeros((self.buffer_size, self.n_samples, self.n_horizen, self.n_action), **self.tensor_args)

        # Sampling class
        self.sample_gen = MixedSampling(self.params, self.tensor_args)

        # base control states
        self.base_pose = Pose()

        # Target states
        self.target_pose = Pose()
        self.target_pose.pose = torch.tensor([0.5, 0.0, 0.6])
        self.target_pose.orientation = torch.tensor([0.0, 0.0, 0.0, 1.0])
        self.predict_target_pose = torch.zeros((self.n_horizen, 6))

        self.diff_ori_3d = torch.zeros((3), **self.tensor_args)
        self.target_dist = torch.zeros((1), **self.tensor_args)

        # MPPI Cost Manager
        self._lambda = params['mppi']['_lambda']
        self.cost_manager = CostManager(self.params, self.tensor_args)

        # Import URDF for forward kinematics
        package_name = params['mppi']['package_name']
        urdf_name = params['mppi']['urdf_name']
        urdf_file_path = os.path.join(get_package_share_directory(package_name), "models", "canadarm", urdf_name)

        # Forward kinematics
        self.fk_canadarm = URDFForwardKinematics(urdf_file_path, root_link='Base_SSRMS', end_links='EE_SSRMS_tip')
        mount_tf = torch.eye(4, **self.tensor_args)
        mount_tf[0:3, 0:3] = euler_angles_to_matrix(torch.tensor([3.1416, 0.0, 0.0]), 'XYZ')
        mount_tf[0:3, 3] = torch.tensor([0.0, 0.0, 3.6])
        self.fk_canadarm.set_mount_transformation(mount_tf)
        self.fk_canadarm.set_samples_and_timesteps(self.n_samples, self.n_horizen, self.n_mobile_dof)

        # Filter
        self.svg_filter = SavGolFilter()

        # Align the target pose
        self.is_align = False

        # Collision avoidance target (debris)
        self.collision_target = torch.tensor([[0.0, 0.0, 0.0]], **self.tensor_args)

        # 250519
        self.noise = None
        self.noise_prev = torch.zeros((self.n_samples, self.n_horizen, self.n_action), **self.tensor_args)
        self.uSamples = None
        self.param_gamma = self._lambda * (1.0 - 0.9)
        self.is_reaching = False
        self.calc_jacob = CanadarmJacob()

        # Log
        self.sim_time = Time()
        self.matlab_logger = MATLABLogger(script_name=Path(__file__).stem, file_name="end_effector_pose")
        self.matlab_logger.create_dataset(dataset_name="end_effector_pose", shape=7)
        self.matlab_logger.create_dataset(dataset_name="pos_err", shape=4)
        self.matlab_logger.create_dataset(dataset_name="ori_err", shape=4)
        self.matlab_logger.create_dataset(dataset_name="cost", shape=11)
        self.matlab_logger.create_dataset(dataset_name="sigma", shape=(self.n_action+1))

        # test
        # from torch.utils.tensorboard import SummaryWriter
        # self.tensor_board_logdir = "/home/user/space_ws/src/mppi_solver/mppi_solver/runs/weights"
        # self.writer = SummaryWriter(log_dir=self.tensor_board_logdir)
        # self.step = 0

        self.reference_joint = None
        self.reference_se3 = None
        self.iteration = 0
        

    def check_reach(self):
        ee_pose_mat, tf_list = self.fk_canadarm.forward_kinematics_cpu(self._q[self.n_mobile_dof:], 
                                'EE_SSRMS_tip', 'Base_SSRMS', init_transformation=self.base_pose.tf_matrix(),
                                free_floating=self.is_free_floating, base_move=self.is_base_move)
        _ = self.calc_jacob.compute_jacobian_cpu(tf_list)      
        self.ee_pose.from_matrix(ee_pose_mat)

        pose_err = pos_diff(self.ee_pose, self.target_pose)
        ee_ori_mat = euler_angles_to_matrix(self.ee_pose.rpy, "ZYX")
        target_ori_mat = euler_angles_to_matrix(self.target_pose.rpy, "ZYX")
        diff_ori_mat = torch.matmul(torch.linalg.inv(ee_ori_mat), target_ori_mat)
        diff_ori_quat = matrix_to_quaternion(diff_ori_mat)
        self.diff_ori_3d = matrix_to_euler_angles(diff_ori_mat, "ZYX")
        self.target_dist = torch.norm(self.ee_pose.pose - self.target_pose.pose, p=2)

        if pose_err < 0.005:
            return True
        else:
            return False
    

    def compute_control_input(self):
        if self.check_reach():
            return self.qdes, self.vdes
        
        self.MATLAB_log()
        
        u = self.u_prev.clone()
        noise = self.sample_gen.sampling(self._q)
        v = u + noise
        qSamples, vSamples = self.sample_gen.get_sample_joint(v, self._q, self._qdot, self.dt)
        qSamples = qSamples.to(**self.tensor_args)
        vSamples = vSamples.to(**self.tensor_args)
        trajectory, link_list, com_list = self.fk_canadarm.forward_kinematics(qSamples, 'EE_SSRMS_tip', 'Base_SSRMS', self.base_pose.tf_matrix(self.tensor_args), 
                                                         free_floating=self.is_free_floating, base_move=self.is_base_move)
        jacob = self.calc_jacob.compute_jacobian(link_list)
        # jacob_bm = self.calc_jacob.compute_jacob_bm(com_list, link_list) 

        self.cost_manager.update_pose_cost(qSamples, v, vSamples, trajectory, self.reference_joint, self.reference_se3, self.target_pose)
        self.cost_manager.update_covar_cost(u, v, self.sample_gen.sigma_matrix)
        self.cost_manager.update_base_cost(self.base_pose, self._q)
        self.cost_manager.update_collision_cost(self.collision_target)
        self.cost_manager.update_ee_cost(jacob, self.target_dist)
        self.cost_manager.update_reference_cost(link_list[-2])
        # self.cost_manager.update_base_disturbance_cost(jacob_bm)

        S = self.cost_manager.compute_all_cost()

        w = self.compute_weights(S, self._lambda)
        # self.logger.info(f"{100*w[0:1024].sum():4}, {100*w[1024:2048].sum():4}, {100*w[2048:3072].sum():4}")
        w_max_idx = torch.argmax(w)
        # self.logger.info(f"widx: {w_max_idx}")
        # self.logger.info(f"w: {w[w_max_idx]}")
        self.logger.info(f"noise: {noise[w_max_idx,:,:]}")
        w_expanded = w.view(-1, 1, 1)
        w_eps = torch.sum(w_expanded * noise, dim = 0)
        w_eps = self.svg_filter.savgol_filter_torch(w_eps, window_size=9, polyorder=2, tensor_args=self.tensor_args)

        u += w_eps

        self.sample_gen.update_distribution(u, v, w, noise)

        self.u_prev = u.clone()
        self.u = u[0].clone()
        self.noise_prev = noise.clone()

        self.vdes = self._qdot + self.u * self.dt
        self.qdes = self._q + self._qddot * self.dt + 0.5 * self.u * self.dt * self.dt

        return self.qdes, self.vdes


    def compute_weights(self, S: torch.Tensor, _lambda: float) -> torch.Tensor:
        z = -S / _lambda
        weights = torch.softmax(z, dim=0)  # (n_samples,)
        return weights
    

    def MATLAB_log(self):
        q_prev, self.v_prev = self.sample_gen.get_prev_sample_joint(self.u_prev, self._q, self._qdot, self.dt)
        self.fk_canadarm.robot._n_samples = 1
        ee_traj_prev, link_list_prev, com_list_prev = self.fk_canadarm.forward_kinematics(q_prev, 'EE_SSRMS_tip', 'Base_SSRMS', \
                       self.base_pose.tf_matrix(self.tensor_args), free_floating=self.is_free_floating, base_move=self.is_base_move)
        ee_traj_prev = ee_traj_prev.squeeze(0).cpu()
        self.fk_canadarm.robot._n_samples = self.n_samples
        ee_jacobian_prev = self.calc_jacob.compute_jacobian(link_list_prev)
        ee_jacobian_prev = ee_jacobian_prev.squeeze(0)

        prev_stage_cost     = self.cost_manager.pose_cost.compute_prev_stage_cost(ee_traj_prev, self.target_pose)
        prev_terminal_cost  = self.cost_manager.pose_cost.compute_prev_terminal_cost(ee_traj_prev, self.target_pose)
        prev_centering_cost = self.cost_manager.joint_cost.compute_prev_centering_cost(q_prev)
        prev_tracking_cost  = self.cost_manager.joint_cost.compute_prev_jointTraj_cost(q_prev, self.reference_joint)
        prev_action_cost    = self.cost_manager.action_cost.compute_prev_action_cost(self.u_prev)
        prev_covar_cost     = self.cost_manager.covar_cost.compute_prev_covar_cost(self.sample_gen.sigma_matrix, self.u_prev, self.noise_prev)
        prev_collision_cost = self.cost_manager.collision_cost.compute_prev_collision_cost(self.base_pose, q_prev, self.collision_target)
        prev_stop_cost      = self.cost_manager.stop_cost.compute_prev_stop_cost(self.v_prev)
        prev_ee_cost        = self.cost_manager.ee_cost.compute_prev_ee_cost(self.v_prev, ee_jacobian_prev, self.target_dist)
        prev_reference_cost = self.cost_manager.reference_cost.compute_prev_reference_cost(link_list_prev[-2], self.reference_se3)
        
        mean_prev_stage_cost     = torch.mean(prev_stage_cost)
        mean_prev_terminal_cost  = torch.mean(prev_terminal_cost)
        mean_prev_centering_cost = torch.mean(prev_centering_cost)
        mean_prev_tracking_cost  = torch.mean(prev_tracking_cost)
        mean_prev_action_cost    = torch.mean(prev_action_cost)
        mean_prev_covar_cost     = torch.mean(prev_covar_cost)
        mean_prev_collision_cost = torch.mean(prev_collision_cost)
        mean_prev_stop_cost      = torch.mean(prev_stop_cost)
        mean_prev_ee_cost        = torch.mean(prev_ee_cost)
        mean_prev_reference_cost = torch.mean(prev_reference_cost)
        
        self.matlab_logger.log("end_effector_pose", [self.sim_time.time] + self.ee_pose.np_pose.tolist() + self.ee_pose.np_rpy.tolist())
        self.matlab_logger.log("pos_err", [self.sim_time.time] + (self.ee_pose.np_pose - self.target_pose.np_pose).tolist())
        self.matlab_logger.log("ori_err", [self.sim_time.time] + self.diff_ori_3d.tolist())
        self.matlab_logger.log("cost", [self.sim_time.time] + [mean_prev_stage_cost.item(),
                                                               mean_prev_terminal_cost.item(),
                                                               mean_prev_centering_cost.item(),
                                                               mean_prev_tracking_cost.item(),
                                                               mean_prev_action_cost.item(),
                                                               mean_prev_covar_cost.item(),
                                                               mean_prev_collision_cost.item(),
                                                               mean_prev_stop_cost.item(),
                                                               mean_prev_ee_cost.item(),
                                                               mean_prev_reference_cost.item()])
        self.matlab_logger.log("sigma", [self.sim_time.time] + torch.diag(self.sample_gen.sigma).tolist())
        return
    

    def set_joint(self, joint_states: torch.Tensor):
        joint_states = joint_states.to(**self.tensor_args)
    
        self._q = joint_states[:, 0]
        self._qdot = joint_states[:, 1]
        self._qddot = joint_states[:, 2]
        return

    def set_target_pose(self, pose: Pose):
        self.target_pose.pose = pose.pose
        self.target_pose.orientation = pose.orientation
        return
    
    def set_base_pose(self, pos, ori):
        self.base_pose.pose = pos
        self.base_pose.orientation = ori
        return
    
    def set_collision_target(self, target: torch.Tensor):
        self.collision_target = target.clone().to(**self.tensor_args)
        return
    
    def setReference(self, reference_joint: torch.Tensor, reference_se3: torch.Tensor):
        self.reference_joint = reference_joint.clone()
        self.reference_se3 = reference_se3.clone()
