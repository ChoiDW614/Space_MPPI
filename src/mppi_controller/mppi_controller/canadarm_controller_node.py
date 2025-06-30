import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile
from rclpy.qos import DurabilityPolicy
from rclpy.qos import ReliabilityPolicy

from std_msgs.msg import Float64MultiArray
from control_msgs.msg import DynamicJointState

import numpy as np
import torch

from mppi_controller.src.wrapper.canadarm_wrapper import CanadarmWrapper


class MppiControllerNode(Node):
    def __init__(self):
        super().__init__("mppi_controller_node")
        self.canadarmWrapper = CanadarmWrapper()

        self.canadaFlag = False
        self.solverFlag= False

        # joint control states
        self.isBaseMoving = False
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

        # model state subscriber
        subscribe_qos_profile = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        self.joint_state_subscriber = self.create_subscription(DynamicJointState, '/dynamic_joint_states', self.joint_state_callback, subscribe_qos_profile)
        self.target_joint_subscriber = self.create_subscription(Float64MultiArray, '/canadarm_joint_controller/target_joint_states', self.target_joint_callback, subscribe_qos_profile)

        # publisher
        cal_timer_period = 0.01  # seconds
        pub_timer_period = 0.01  # seconds
        self.cal_timer = self.create_timer(cal_timer_period, self.cal_timer_callback)
        self.pub_timer = self.create_timer(pub_timer_period, self.pub_timer_callback)

        self.arm_msg = Float64MultiArray()
        if self.isBaseMoving:
            self.arm_publisher = self.create_publisher(Float64MultiArray, '/floating_canadarm_joint_controller/commands', 10)
        else:
            self.arm_publisher = self.create_publisher(Float64MultiArray, '/canadarm_joint_controller/commands', 10)


    def cal_timer_callback(self):
        if self.canadaFlag:
            if not self.solverFlag:
                qdes = np.array([0.0, -0.0, 0.0, -0.0, 0.0, 0.0, 0.0])
                qddot_des = 400 * (qdes - self.canadarmWrapper.state.q) - 10 * self.canadarmWrapper.state.v
                u = self.canadarmWrapper.state.M @ qddot_des + self.canadarmWrapper.state.G
                self.arm_msg.data = u.tolist()
            else:
                # self.controller.set_joint(self.interface_values)
                # u, qdes, vdes = self.controller.compute_control_input()
                # qdes = qdes.clone().cpu().numpy()
                # vdes = vdes.clone().cpu().numpy()

                # qddot_des = 40 * (qdes - self.canadarmWrapper.state.q)  + 4 * (vdes - self.canadarmWrapper.state.v)
                # qddot_des = u.clone().cpu().numpy()
                target_joint = np.array(self.target_joint)
                qddot_des = 400 * (target_joint[:7] - self.canadarmWrapper.state.q) + 40 * (target_joint[7:] - self.canadarmWrapper.state.v)
                u = self.canadarmWrapper.state.M @ qddot_des + self.canadarmWrapper.state.G
                self.arm_msg.data = u.tolist()
        return


    def pub_timer_callback(self):
        if self.canadaFlag and self.solverFlag:
            self.arm_publisher.publish(self.arm_msg)
        return


    def joint_state_callback(self, msg):
        self.canadaFlag = True
        self.joint_names = msg.joint_names
        self.interface_name = [iv.interface_names for iv in msg.interface_values]
        values = [list(iv.values) for iv in msg.interface_values]

        index_map = [self.joint_names.index(joint) for joint in self.joint_order]
        self.interface_values = torch.tensor([values[i] for i in index_map])
        self.canadarmWrapper.state.q = self.interface_values.clone().cpu().numpy()[:,0]
        self.canadarmWrapper.state.v = self.interface_values.clone().cpu().numpy()[:,1]
        self.canadarmWrapper.computeAllTerms()
        return


    def target_joint_callback(self, msg):

        self.solverFlag = True
        self.target_joint = msg.data
                
        return



def main():
    rclpy.init()
    node = MppiControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
