import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile
from rclpy.qos import DurabilityPolicy
from rclpy.qos import ReliabilityPolicy

from control_msgs.msg import DynamicJointState
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image

from rclpy.logging import get_logger

import numpy as np
import torch

from mppi_solver.src.solver.mppi_franka import MPPI

from mppi_solver.src.utils.pose import Pose
# from mppi_solver.src.utils.image_pipeline import ros_to_cv2


class MppiSolverNode(Node):
    def __init__(self):
        super().__init__("mppi_solver_node")

        # target states
        init_interface_pose = Pose()
        

        self.joint_order = [
            "panda_joint1","panda_joint2","panda_joint3","panda_joint4","panda_joint5","panda_joint6","panda_joint7"]
        self.joint_names = None
        self.interface_name = None
        self.interface_values = None

        self.qdes = np.zeros(7)
        self.vdes = np.zeros(7)

        # controller
        self.controller = MPPI()

        # model state subscriber
        subscribe_qos_profile = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        
        self.joint_state_subscriber = self.create_subscription(DynamicJointState, '/dynamic_joint_states', self.joint_state_callback, subscribe_qos_profile)
    
        # publisher
        cal_timer_period = 0.01  # seconds
        pub_timer_period = 0.01  # seconds
        self.cal_timer = self.create_timer(cal_timer_period, self.cal_timer_callback)
        self.pub_timer = self.create_timer(pub_timer_period, self.pub_timer_callback)

        self.arm_msg = Float64MultiArray()
        self.arm_publisher = self.create_publisher(Float64MultiArray, '/franka_joint_controller/target_joint_states', 10)


    def cal_timer_callback(self):
        qdes, vdes = self.controller.compute_control_input()
        self.qdes = qdes.clone().cpu().numpy()
        self.vdes = vdes.clone().cpu().numpy()
        return


    def pub_timer_callback(self):
        self.arm_msg.data =[]
        for i in range(0,7):
            self.arm_msg.data.append(self.qdes[i])
        for i in range(0,7):
            self.arm_msg.data.append(self.vdes[i])
        self.arm_publisher.publish(self.arm_msg)
        return


    def joint_state_callback(self, msg):
        self.joint_names = msg.joint_names
        self.interface_name = [iv.interface_names for iv in msg.interface_values]
        values = [list(iv.values) for iv in msg.interface_values]

        index_map = [self.joint_names.index(joint) for joint in self.joint_order]
        self.interface_values = torch.tensor([values[i] for i in index_map])
        self.controller.set_joint(self.interface_values)
        return
    

def main():
    rclpy.init()
    node = MppiSolverNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
