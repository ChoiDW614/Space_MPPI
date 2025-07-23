import os

import torch
import numpy as np
from torch.linalg import inv

from mppi_solver.src.utils.pose import Pose

from rclpy.logging import get_logger
from ament_index_python.packages import get_package_share_directory


from mppi_solver.src.trajectory.trajManager import SE3Traj
from mppi_solver.src.wrapper.canadarm_wrapper import CanadarmWrapper
from pinocchio.utils import *
import pinocchio as pin 

from copy import deepcopy

class IKSolver:
    def __init__(self, params):
        self.logger = get_logger("IKSolver")
        self.se3_traj = SE3Traj()
        
        self.duration = None
        self.mount_tf = torch.eye((4))
    
        self.initSE3 = pin.SE3(1)

        self.robot = CanadarmWrapper(params)
        self.pgain = 100.0
        self.k_null = 0.1
        self.prev_trajectory = np.zeros((32, 7))


    def forward_kinematics(self, q, base_pose: Pose):
        self.full_q[1:8] = q.tolist()
        frames = self.chain.forward_kinematics(self.full_q, full_kinematics=True)
        self.local_ee_pose.from_matrix(torch.tensor(frames[8]))
        self.ee_pose.from_matrix(base_pose.tf_matrix() @ self.mount_tf @ self.local_ee_pose.tf_matrix())
        return self.ee_pose


    def inverse_kinematics(self, target_pose: Pose, base_pose: Pose, init_states: torch.Tensor=None):
        target_frame = target_pose.tf_matrix()
        ee_local_target = inv(self.mount_tf) @ inv(base_pose.tf_matrix()) @ target_frame

        initial_joints = self.full_initial_joints
        if init_states is not None:
            initial_joints[1:8] = init_states.tolist()

        joint_angles_full = self.chain.inverse_kinematics_frame(
            ee_local_target,
            initial_joints
        )
        joint_states = torch.tensor(joint_angles_full[1:8])
        
        return joint_states
    

    def init_ik_trajectory(self, duration, stime, initSE3,targetSE3):
        self.se3_traj.setStartTime(stime)
        self.se3_traj.setDuration(duration)
        self.se3_traj.setInitSample(initSE3)
        self.se3_traj.setTargetSample(targetSE3)
        self.initSE3 = deepcopy(initSE3)
        

    def get_ik_trajectory(self, ctime, timewindow, dt):
        poseTraj = torch.zeros((timewindow, 4, 4))

        for i in range(timewindow):
            c_time = ctime + i * dt
            self.se3_traj.setCurrentTime(c_time)
            se3_cubic = self.se3_traj.computeNext()
            se3_pose = pin.SE3ToXYZQUAT(se3_cubic)
            se3_tensor = torch.tensor(se3_cubic.homogeneous)

            pose = Pose()
            pose.pose = torch.tensor(se3_pose[:3])
            pose.orientation = torch.tensor(se3_pose[3:])
            # pose_list.append(pose)
            poseTraj[i, :, :] = se3_tensor.clone()

        return poseTraj

    def get_ik_joint_trajectory(self, ctime, oMi_current, q_current, timewindow, dt):
        
        vel_traj = np.zeros((timewindow, 6))
        torch_joint_traj = torch.zeros((timewindow, self.robot.model.nq))
        pose_traj = []
        for i in range(timewindow):
            c_time = ctime + i * dt
            self.se3_traj.setCurrentTime(c_time)
            se3_cubic = self.se3_traj.computeNext()
            pose_traj.append(se3_cubic)
            if i==0:
                del_se3 = oMi_current.inverse() * se3_cubic
            else:
                del_se3 = pose_traj[i-1].inverse() * se3_cubic
            
            vel_traj[i,:] = self.pgain * pin.log(del_se3).vector
        joint_traj = [q_current]
        for i in range(timewindow):
            self.robot.state.q = joint_traj[i]
            self.robot.state.v = np.zeros((self.robot.model.nq))
            self.robot.computeAllTerms()
            qdot = np.linalg.pinv(self.robot.state.J) @ vel_traj[i,:]
            joint_traj.append(joint_traj[i] + qdot * dt)
            torch_joint_traj[i,:] = torch.tensor(joint_traj[i] + qdot * dt)
        
        return torch_joint_traj
    

    def get_ik_joint_trajectory2(self, ctime, oMi_current, q_current, timewindow, dt):
        vel_traj = np.zeros((timewindow, 6))
        torch_joint_traj = torch.zeros((timewindow, self.robot.model.nq))
        torch_pose_traj = torch.empty((timewindow, 4, 4))
        pose_traj = []
        for i in range(timewindow):
            c_time = ctime + i * dt
            self.se3_traj.setCurrentTime(c_time)
            se3_cubic = self.se3_traj.computeNext()
            pose_traj.append(se3_cubic)
            torch_pose_traj[i,:,:] = torch.from_numpy(pose_traj[i].homogeneous)
            if i==0:
                del_se3 = oMi_current.inverse() * se3_cubic
            else:
                del_se3 = pose_traj[i-1].inverse() * se3_cubic
            
            vel_traj[i,:] = self.pgain * pin.log(del_se3).vector

        joint_traj = [q_current]
        for i in range(timewindow):
            q = joint_traj[-1]
            self.robot.state.q = q
            self.robot.state.v = np.zeros((self.robot.model.nq))
            self.robot.computeAllTerms()
            J = self.robot.state.J
            qdot_p = np.linalg.pinv(J) @ vel_traj[i]
            N = np.eye(self.robot.model.nq) - np.linalg.pinv(J) @ J
            qdot_s = N @ (self.prev_trajectory[i,:] - q)
            q_next = q + (qdot_p + self.k_null * qdot_s) * dt

            q_next = (q_next + np.pi) % (2 * np.pi) - np.pi

            joint_traj.append(q_next)
            torch_joint_traj[i,:] = torch.tensor(q_next)

        return torch_joint_traj, torch_pose_traj
    

    def targetUpdate(self, target):
        self.se3_traj.setTargetSample(target)
    # def get_ik_joint_trajectory(self, ctime, oMi_current,q_current, timewindow, dt):
    #     vel_traj = []
    #     pose_traj = []
    #     i = 0
    #     while True:
    #         c_time = ctime + i * dt
    #         self.se3_traj.setCurrentTime(c_time)
    #         se3_cubic = self.se3_traj.computeNext()
    #         pose_traj.append(se3_cubic)
    #         if i==0:
    #             del_se3 = oMi_current.inverse() * se3_cubic
    #         else:
    #             del_se3 = pose_traj[i-1].inverse() * se3_cubic
            
    #         vel_traj.append(100 * pin.log(del_se3).vector)
    #         self.logger.warn(f"del se3 {se3_cubic}")

    #         error = np.linalg.norm(pin.log(se3_cubic.inverse() * self.se3_traj.target).vector)
    #         if error<1e-3:
    #             self.logger.info(f"Iter : {i}")
    #             break
    #         i += 1
    #     # vel_traj = [np.array(vel) for vel in vel_traj]
    #     # vel_traj = np.array(vel_traj)  # shape: (N, 6)

    #     # self.logger.info(f"Shape : {vel_traj.shape}")

    #     joint_traj = [q_current]
    #     torch_joint_traj = torch.zeros((len(vel_traj), self.robot.model.nq))

    #     for i in range(len(vel_traj)):
    #         self.robot.state.q = joint_traj[i]
    #         self.robot.state.v = np.zeros((self.robot.model.nq))
    #         self.robot.computeAllTerms()

    #         qdot = np.linalg.pinv(self.robot.state.J) @ vel_traj[i]
    #         # self.logger.info(f"J : {np.linalg.det(self.robot.state.J.T @ self.robot.state.J)}")
    #         # self.logger.info(f"Vel : {vel_traj[i]}")
    #         # self.logger.info(f"qdot : {qdot}")      
    #         # self.logger.info(f"state : {self.robot.state.q}")      
    #         # self.logger.info(f"ans : {self.robot.state.q.copy() + qdot * dt}")
    #         joint_traj.append(self.robot.state.q.copy() + qdot * dt)
    #         torch_joint_traj[i,:] = torch.tensor(self.robot.state.q.copy() + qdot * dt)
    #         # self.logger.info(f"tmp : {self.robot.state.q.copy() + qdot * dt}")
        
    #     # self.logger.info(f"Joint Traj :{torch_joint_traj}")
        
    #     return torch_joint_traj