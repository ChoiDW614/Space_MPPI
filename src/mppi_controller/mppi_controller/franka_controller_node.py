import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile
from rclpy.qos import DurabilityPolicy
from rclpy.qos import ReliabilityPolicy

from std_msgs.msg import Float64MultiArray
from control_msgs.msg import DynamicJointState

import numpy as np
import torch

from mppi_controller.src.wrapper.franka_wrapper import FrankaWrapper


class MppiControllerNode(Node):
    def __init__(self):
        super().__init__("mppi_controller_node")
        self.frankaWrapper = FrankaWrapper()

        self.frankaFlag = False
        self.solverFlag= False

        self.joint_order = [
            "panda_joint1","panda_joint2","panda_joint3","panda_joint4","panda_joint5","panda_joint6","panda_joint7"]
        self.joint_names = None
        self.interface_name = None
        self.interface_values = None

        # model state subscriber
        subscribe_qos_profile = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        
        self.joint_state_subscriber = self.create_subscription(DynamicJointState, '/dynamic_joint_states', self.joint_state_callback, subscribe_qos_profile)
        self.target_joint_subscriber = self.create_subscription(Float64MultiArray, '/franka_joint_controller/target_joint_states', self.target_joint_callback, subscribe_qos_profile)

        cal_timer_period = 0.02  # seconds
        pub_timer_period = 0.001  # seconds
        self.cal_timer = self.create_timer(cal_timer_period, self.cal_timer_callback)
        self.pub_timer = self.create_timer(pub_timer_period, self.pub_timer_callback)

        # arm publisher
        self.arm_msg = Float64MultiArray()
        self.arm_publisher = self.create_publisher(Float64MultiArray, '/franka_joint_controller/commands', 10)

    
    def cal_timer_callback(self):
        if self.frankaFlag:
            if not self.solverFlag:
                qdes = np.array([0.0, -0.78, 0.0, -2.35, 0.0, 1.57, 0.0])
                qddot_des = 40 * (qdes - self.frankaWrapper.state.q) - 10 * self.frankaWrapper.state.v
                u = self.frankaWrapper.state.M @ qddot_des + self.frankaWrapper.state.G
                self.arm_msg.data = u.tolist()
                self.get_logger().info("Here ")
            else :
                # qdes = qdes.clone().cpu().numpy()
                # vdes = vdes.clone().cpu().numpy()

                # qddot_des = 40 * (qdes - self.frankaWrapper.state.q)  + 4 * (vdes - self.frankaWrapper.state.v)
                # qddot_des = u.clone().cpu().numpy()
                test1 = np.array(self.test)
                qddot_des = 100 * (test1[:7] - self.frankaWrapper.state.q) + 10 * (test1[7:] - self.frankaWrapper.state.v)
                u = self.frankaWrapper.state.M @ qddot_des + self.frankaWrapper.state.G
                self.arm_msg.data = u.tolist()
        return


    def pub_timer_callback(self):
        if self.frankaFlag:
            self.arm_publisher.publish(self.arm_msg)
        return


    def joint_state_callback(self, msg):
        self.frankaFlag = True
        self.joint_names = msg.joint_names
        self.interface_name = [iv.interface_names for iv in msg.interface_values]
        values = [list(iv.values) for iv in msg.interface_values]

        index_map = [self.joint_names.index(joint) for joint in self.joint_order]
        self.interface_values = torch.tensor([values[i] for i in index_map])

        self.frankaWrapper.state.q = self.interface_values.clone().cpu().numpy()[:,0]
        self.frankaWrapper.state.v = self.interface_values.clone().cpu().numpy()[:,1]
        self.frankaWrapper.computeAllTerms()
        return


    def target_joint_callback(self, msg):
        self.solverFlag = True
        self.test = msg.data
        return


def main():
    rclpy.init()
    node = MppiControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()



#########TEST
# class PyNode(Node):
#     def __init__(self):
#         super().__init__("robot_node")
#         # IGNITION TEST #
        
#         # self.canadarmWrapper = CanadarmWrapper()
#         self.controller = MPPI()

#         self.target_pose = Pose()

#         self.test = None

#         self.joint_names = None
#         self.interface_name = None
#         self.interface_values = None
#         self.joint_order = [
#             "panda_joint1","panda_joint2","panda_joint3","panda_joint4","panda_joint5","panda_joint6","panda_joint7"
#         ]
#         # model state subscriber
#         subscribe_qos_profile = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        
#         self.joint_state_subscriber = self.create_subscription(DynamicJointState, '/dynamic_joint_states', self.joint_state_callback, subscribe_qos_profile)


#         # IGNITION TEST #
#         self.arm_publisher = self.create_publisher(Float64MultiArray, '/franka_joint_controller/commands', 10)
#         # self.publisher_ = self.create_publisher(JointState, 'franka_joint_command', 10)

#         self.q_ = np.zeros(7)
#         self.v_ = np.zeros(7)

#         #### IGNITION TEST ####
#         def joint_state_callback(self, msg):
#             self.joint_names = msg.joint_names
#             self.interface_name = [iv.interface_names for iv in msg.interface_values]
#             values = [list(iv.values) for iv in msg.interface_values]

#             index_map = [self.joint_names.index(joint) for joint in self.joint_order]
#             self.interface_values = torch.tensor([values[i] for i in index_map])

#             self.q_ = self.interface_values[:,0].clone().cpu().numpy()
#             self.v_ = self.interface_values[:,1].clone().cpu().numpy()

                
#         #### IGNITION TEST ####
#         def jointPublish(self, u):
#             # q : nparray
#             msg = Float64MultiArray()
#             msg.data = u.tolist()
            
#             self.arm_publisher.publish(msg)



# def main():
#     rclpy.init(args = None)
#     node = PyNode()

#     frankaWrapper = FrankaWrapper()
#     controller = MPPI()


#     iter = 0
#     try:
#         while rclpy.ok():
#             frankaWrapper.state.q = np.copy(node.q_)
#             frankaWrapper.state.v = np.copy(node.v_)
#             frankaWrapper.computeAllTerms()

#             if iter < 300:
#                 qdes = np.array([0.0, -0.78, 0.0, -2.35, 0.0, 1.57, 0.0])
#                 qddot_des = 40 * (qdes - self.frankaWrapper.state.q) - 10 * self.frankaWrapper.state.v
#                 u = frankaWrapper.state.M @ qddot_des + self.frankaWrapper.state.G
                

#             else :
#                 controller.set_joint(self.interface_values)
#                 u, qdes, vdes = self.controller.compute_control_input()
#                 qdes = qdes.clone().cpu().numpy()
#                 vdes = vdes.clone().cpu().numpy()

#                 qddot_des = 40 * (qdes - self.frankaWrapper.state.q)  + 4 * (vdes - self.frankaWrapper.state.v)
#                 qddot_des = u.clone().cpu().numpy()
#                 u = self.frankaWrapper.state.M @ qddot_des + self.frankaWrapper.state.G
                
                
#             node.jointPublish(u)

#                 # iter += 1













#                 # print("target : ",oMg.translation)
#                 # print("Current : ",oMi.translation)
#                 # solver.qInit(node.q_, node.v_)
#                 # u, qd, vd = solver.computeNext()
#                 # qd = qd.to("cpu")
#                 # qd = qd.numpy()
#                 # vd = vd.cpu().numpy()
#                 # # print("Qd :", qd )


#                 # node.jointPublish(qd, vd)
#                 # iter2 += 1

#                 # ctime = time.time()

#                 # if iter2 == 200:
#                 #     print("Control End")
#                 #     controlFlag_ = False
#                 #     initFlag_ = False
#                 #     iter2 = 0

#             rclpy.spin_once(node, timeout_sec=0.01)



#     except KeyboardInterrupt:
#         print("Node Cancel")


# if __name__ == "__main__":
#     main()