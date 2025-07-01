# Python imports
import os
import yaml
import math
import time
from copy import deepcopy

# Linear Algebra
import numpy as np
import torch

# RCLPY
import rclpy
from rclpy.node import Node
from rclpy._rclpy_pybind11 import RCLError
from ament_index_python.packages import get_package_share_directory

from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import DurabilityPolicy
from rclpy.qos import ReliabilityPolicy

# ROS2 Messages
from control_msgs.msg import DynamicJointState
from geometry_msgs.msg import TransformStamped, PoseStamped
from std_msgs.msg import Float64MultiArray, Bool
from rosgraph_msgs.msg import Clock

# MPPI Solver Library
from mppi_solver.src.solver.mppi_canadarm import MPPI
from mppi_solver.src.solver.target.kalman_target_state import DockingInterface
from mppi_solver.src.utils.time import Time

from mppi_solver.src.utils.pose import Pose
from mppi_solver.src.utils.rotation_conversions import matrix_to_euler_angles

from mppi_solver.src.wrapper.canadarm_wrapper import CanadarmWrapper
from mppi_solver.src.trajectory.trajManager import SE3Traj
from mppi_solver.src.robot.ik.canadarm_ik import IKSolver

# Pinocchio Library
import pinocchio as pin
from pinocchio.utils import *

# Logger : MATLAB for Plot
from pathlib import Path
from mppi_solver.src.utils.matlab_logger import MATLABLogger


class MppiSolverNode(Node):
    def __init__(self):
        super().__init__("mppi_solver_node")

        # Load Yaml Config
        self.package_name = "mppi_solver"
        config_path = os.path.join(get_package_share_directory(self.package_name), "configs", "canadarm", "reach.yaml")
        try:
            with open(config_path, "r") as file:
                params = yaml.load(file, Loader=yaml.FullLoader)
        except FileNotFoundError:
            self.get_logger().error(f"Config file not found: {config_path}")
            raise RCLError(f"Config file not found: {config_path}")

        # ROBOT
        self.canadarmWrapper = CanadarmWrapper()

        # joint control states
        self.isBaseMoving = params['mppi']['isBaseMoving']
        if self.isBaseMoving:
            self.joint_order = [
                "v_x_joint", "v_y_joint", "v_z_joint", "v_r_joint", "v_p_joint", "v_yaw_joint",
                "Base_Joint", "Shoulder_Roll", "Shoulder_Yaw", "Elbow_Pitch", "Wrist_Pitch", "Wrist_Yaw", "Wrist_Roll"]
        else:
            self.joint_order = [
                "Base_Joint", "Shoulder_Roll", "Shoulder_Yaw", "Elbow_Pitch", "Wrist_Pitch", "Wrist_Yaw", "Wrist_Roll"]
        self.joint_names = None
        self.interface_name = None
        self.interface_values = None
        self.qdes = np.zeros(7)
        self.vdes = np.zeros(7)

        # controller
        self.controller = MPPI(params)

        # target states
        self.is_sim_ros2_connected = False
        self.docking_interface = DockingInterface(self.controller, predict_step=32)
        self.canadarmIK = IKSolver(self.isBaseMoving)
        self.sim_time = Time()

        # model state subscriber
        subscribe_qos_profile = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        subscribe_qos_profile2 = QoSProfile(history=QoSHistoryPolicy.KEEP_ALL, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        
        self.joint_state_subscriber = self.create_subscription(DynamicJointState, '/dynamic_joint_states', self.joint_state_callback, subscribe_qos_profile)
        self.base_state_subscriber = self.create_subscription(TransformStamped, '/model/canadarm/pose', self.model_state_callback, subscribe_qos_profile2)
        self.target_state_subscriber = self.create_subscription(TransformStamped, '/model/ets_vii/pose', self.target_state_callback, subscribe_qos_profile)
        self.sim_clock_subscriber = self.create_subscription(Clock, '/world/default/clock', self.sim_clock_callback, subscribe_qos_profile)

        # publisher
        cal_timer_period = 0.01  # seconds
        pub_timer_period = 0.01  # seconds
        self.cal_timer = self.create_timer(cal_timer_period, self.cal_timer_callback)
        self.pub_timer = self.create_timer(pub_timer_period, self.pub_timer_callback)

        self.arm_msg = Float64MultiArray()
        self.arm_publisher = self.create_publisher(Float64MultiArray, '/canadarm_joint_controller/target_joint_states', 10)

        # Log
        self.matlab_logger = MATLABLogger(script_name=Path(__file__).stem, file_name="joint_states")
        self.matlab_logger.create_dataset(dataset_name="joint_states", shape=8)

        # Controller MSG
        self.arm_msg = Float64MultiArray()
        self.arm_publisher = self.create_publisher(Float64MultiArray, '/canadarm_joint_controller/target_joint_states', 10)
        
        self.is_reaching = False
        self.is_init_trajectory = True
        self.target = Pose()
        self.traj_se3 = SE3Traj()
        self.init_jointCB = False
        self.is_target = False

        # Compute Joint Traj
        self.targetSE3 : None

        # TEST
        self.tmp = None


    def cal_timer_callback(self):
        if self.is_target and self.init_jointCB and (self.sim_time.time > 10.0):
        # if self.is_target and self.init_jointCB:
            if self.is_init_trajectory:

                targetSE3 = self.targetSE3 * self.canadarmWrapper.eef_to_tip.inverse()
                stime = time.time()
                init_oMi = deepcopy(self.canadarmWrapper.iss_to_base * self.canadarmWrapper.state.oMi * self.canadarmWrapper.eef_to_tip)
                self.canadarmIK.init_ik_trajectory(5.0, stime, init_oMi, targetSE3)
                # self.tmp = self.canadarmIK.get_ik_joint_trajectory(stime, init_oMi,self.canadarmWrapper.state.q.copy(), 32, 0.01)
                self.is_init_trajectory = False

            self.canadarmIK.targetUpdate(self.targetSE3 * self.canadarmWrapper.eef_to_tip.inverse())
            oMi = self.canadarmWrapper.iss_to_base * self.canadarmWrapper.state.oMi
            ctime = time.time()
            jointTraj = self.canadarmIK.get_ik_joint_trajectory(ctime, oMi, self.canadarmWrapper.state.q.copy(), 32, 0.01)
            self.controller.setReference(jointTraj)
            qdes, vdes = self.controller.compute_control_input()
            self.qdes = qdes.clone().cpu().numpy()
            self.vdes = vdes.clone().cpu().numpy()
        return


    def pub_timer_callback(self):
        if self.is_sim_ros2_connected:
            self.arm_msg.data =[]
            for i in range(0,7):
                self.arm_msg.data.append(self.qdes[i])
            for i in range(0,7):
                self.arm_msg.data.append(self.vdes[i])

            for i, x in enumerate(self.arm_msg.data):
                if isinstance(x, float) and math.isnan(x):
                    self.arm_msg.data[i] = 0.0

            self.arm_publisher.publish(self.arm_msg)
        return


    def joint_state_callback(self, msg):
        if self.is_sim_ros2_connected:
            self.joint_names = msg.joint_names
            self.interface_name = [iv.interface_names for iv in msg.interface_values]
            values = [list(iv.values) for iv in msg.interface_values]

            index_map = [self.joint_names.index(joint) for joint in self.joint_order]
            self.interface_values = torch.tensor([values[i] for i in index_map])
            self.controller.set_joint(self.interface_values)

            # ROBOT UPDATE
            self.canadarmWrapper.state.q = self.interface_values.clone().cpu().numpy()[:,0]
            self.canadarmWrapper.state.v = self.interface_values.clone().cpu().numpy()[:,1]
            self.canadarmWrapper.computeAllTerms()
            self.init_jointCB = True
            self.matlab_logger.log("joint_states", [self.docking_interface.sim_time.time] + self.interface_values[:, 0].tolist())
        return
    

    def target_state_callback(self, msg):
        if not self.is_sim_ros2_connected:
            if msg.child_frame_id == 'ets_vii':
                self.docking_interface.set_true_docking_pose(msg)
                if torch.allclose(self.docking_interface.true_docking_pose.pose, torch.tensor([0.0, 0.0, 0.0])) \
                    and torch.allclose(self.docking_interface.true_docking_pose.orientation, torch.tensor([0.0, 0.0, 0.0, 1.0])):
                    return
                else:
                    self.is_sim_ros2_connected = True
        else:
            if msg.child_frame_id == 'ets_vii':
                self.docking_interface.set_true_docking_pose(msg)

                # calculate velocity
                self.docking_interface.update_velocity()

                # kalman filter update
                self.docking_interface.ekf_update()

                self.controller.set_target_pose(self.docking_interface.true_align_docking_pose)
                # self.controller.set_predict_target_pose(self.docking_interface.predict_pose)

                self.target = deepcopy(self.docking_interface.true_align_docking_pose)
                self.targetSE3 = pin.XYZQUATToSE3(np.array([self.target.pose[0], self.target.pose[1], self.target.pose[2], self.target.orientation[0], self.target.orientation[1], self.target.orientation[2], self.target.orientation[3]]))
                # self.get_logger().info(f"Target : {self.targetSE3}")

                # prev state
                self.docking_interface.pose_prev = self.docking_interface.pose
                self.docking_interface.time_prev = self.docking_interface.time

                self.is_target = True
        return
    

    def model_state_callback(self, msg):
        if msg.header.frame_id == "default":
            if msg.child_frame_id == "canadarm":
                self.controller.set_base_pose(msg.transform.translation, msg.transform.rotation)
        return
    
    def sim_clock_callback(self, msg):
        self.docking_interface.sim_time.time = msg.clock
        self.controller.sim_time.time = msg.clock
        self.sim_time.time = msg.clock
        return
    

def main():
    rclpy.init()
    node = MppiSolverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.docking_interface.matlab_logger.close()
            node.controller.matlab_logger.close()
        except Exception as e:
            node.get_logger().error(f"Failed to close log file: {e}")
        node.destroy_node()
        try:
            rclpy.shutdown()
        except:
            pass
