import os
import numpy as np
from numpy.linalg import pinv

from rclpy.logging import get_logger

import pinocchio as pin
from pinocchio import RobotWrapper
from pinocchio.utils import *

from ament_index_python.packages import get_package_share_directory


class state():
    def __init__(self):
        self.q: np.array
        self.v: np.array
        self.a: np.array
        self.q_des: np.array
        self.v_des: np.array
        self.a_des: np.array
        self.q_ref: np.array
        self.v_ref: np.array

        self.acc: np.array
        self.tau: np.array
        self.torque: np.array
        self.v_input: np.array

        self.nq: np.array
        self.nv: np.array
        self.na: np.array

        self.id: np.array
        self.G: np.array
        self.M: np.array
        self.J: np.array
        self.M_q: np.array

        self.oMi : pin.SE3


class CanadarmWrapper(RobotWrapper):
    def __init__(self):
        package_name = "mppi_controller"
        urdf_file_path = os.path.join(get_package_share_directory(package_name), "models", "canadarm", "Canadarm2_w_iss.urdf")
        self.__robot = self.BuildFromURDF(urdf_file_path)

        self.data, self.__collision_data, self.__visual_data = \
            pin.createDatas(self.__robot.model, self.__robot.collision_model, self.__robot.visual_model)
        self.model = self.__robot.model
        self.model.gravity = pin.Motion.Zero()

        self.state = state()

        self.__ee_joint_name = "Wrist_Roll"
        self.state.id = self.index(self.__ee_joint_name)

        self.state.nq = self.__robot.nq
        self.state.nv = self.__robot.nv
        self.state.na = self.__robot.nv

        self.state.q = zero(self.state.nq)
        self.state.v = zero(self.state.nv)
        self.state.a = zero(self.state.na)
        self.state.acc = zero(self.state.na)
        self.state.tau = zero(self.state.nv)

        self.state.oMi = pin.SE3()
        self.eef_to_tip = pin.XYZQUATToSE3(np.array([0,0,-1.4, 1, 0, 0, 0]))


    def computeAllTerms(self):
        pin.computeAllTerms(self.model, self.data, self.state.q, self.state.v)
        self.computeJointJacobians(self.state.q)
        self.state.G = self.nle(self.state.q, self.state.v)     # NonLinearEffects
        self.state.M = self.mass(self.state.q)                  # Mass
        self.state.a = pin.aba(self.model, self.data, self.state.q, self.state.v, self.state.tau)
        self.state.oMi = self.data.oMi[self.state.id]
        Adj_mat = self.computeAdjMat(self.eef_to_tip)
        self.state.J = self.getJointJacobian(self.state.id)

    def computeAdjMat(self, aTb : pin.SE3):
        rot = aTb.rotation
        trans = aTb.translation

        Adj = np.zeros((6,6))
        Adj[:3, :3] = Adj[3:, 3:] = rot.copy()
        trans_skew = np.array([[0, -trans[2], trans[1]],[trans[2], 0, -trans[1]],[-trans[1], trans[0],0]])
        Adj[:3, 3:] = trans_skew @ rot

        return Adj
