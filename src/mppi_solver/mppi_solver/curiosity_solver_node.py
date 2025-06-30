import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile
from rclpy.qos import HistoryPolicy
from rclpy.qos import DurabilityPolicy
from rclpy.qos import ReliabilityPolicy


from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, PoseStamped, Twist
from visualization_msgs.msg import Marker, MarkerArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from nav_msgs.msg import Odometry
from control_msgs.msg import DynamicJointState

from gazebo_msgs.srv import SetEntityState
from std_srvs.srv import Empty

import math
import time
import numpy as np
import torch
import functools
import random
from typing import Tuple

from mppi_solver.src.solver.mppi_curiosity import MPPI
from mppi_solver.src.robot.curiosity_wrapper import CuriosityWrapper


class header():
    def __init__(self):
        self.frame_id = str()
        self.stamp = None


class header():
    def __init__(self):
        self.frame_id = str()
        self.stamp = None


class mppiControllerNode(Node):
    def __init__(self):
        super().__init__("mppi_solver_node")

        self.curiosityWrapper = CuriosityWrapper()

        self.controller = MPPI()

        self.header = header()
        self.base_pose = torch.zeros(7)
        self.base_twist = torch.zeros(6)

        # model state subscriber
        subscribe_qos_profile = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        
        self.joint_state_subscriber = self.create_subscription(DynamicJointState, '/dynamic_joint_states', self.joint_state_callback, subscribe_qos_profile)
        self.base_state_subscriber = self.create_subscription(Odometry, '/model/curiosity_mars_rover/odometry', self.base_state_callback, subscribe_qos_profile)

        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # arm publisher
        self.arm_publisher_ = self.create_publisher(Float64MultiArray, '/arm_joint_effort_controller/commands', 10)

        # mast publisher
        self.mast_publisher_ = self.create_publisher(JointTrajectory, '/mast_joint_trajectory_controller/joint_trajectory', 10)

        # motion publisher
        self.curr_action = Twist()
        self.motion_publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
    
    
    def timer_callback(self):
        self.controller.compute_control()
        self.motion_publisher_.publish(self.curr_action)

    def joint_state_callback(self, msg):
        self.joint_names = msg.joint_names
        self.interface_name = msg.interface_values
        self.interface_values = msg.interface_values

        # self.get_logger().info(str(self.joint_names))
        self.get_logger().info(str(self.interface_name))
        self.get_logger().info(str(self.interface_values))

    
    def base_state_callback(self, msg):
        self.header.frame_id = msg.header.frame_id
        self.header.stamp = msg.header.stamp
        self.child_frame_id = msg.child_frame_id
        self.base_pose[0] = msg.pose.pose.position.x
        self.base_pose[1] = msg.pose.pose.position.y
        self.base_pose[2] = msg.pose.pose.position.z
        self.base_pose[3] = msg.pose.pose.orientation.x
        self.base_pose[4] = msg.pose.pose.orientation.y
        self.base_pose[5] = msg.pose.pose.orientation.z
        self.base_pose[6] = msg.pose.pose.orientation.w
        self.base_twist[0] = msg.twist.twist.linear.x
        self.base_twist[1] = msg.twist.twist.linear.y
        self.base_twist[2] = msg.twist.twist.linear.z
        self.base_twist[3] = msg.twist.twist.angular.x
        self.base_twist[4] = msg.twist.twist.angular.y
        self.base_twist[5] = msg.twist.twist.angular.z


def main():
    rclpy.init()
    node = mppiControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    