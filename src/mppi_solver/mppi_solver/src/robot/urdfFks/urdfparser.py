"""This module is in most parts copied from https://github.com/mahaarbo/urdf2casadi.

Changes are in get_forward_kinematics as it allows to pass the variable as an argument.
"""
# import casadi as ca
import numpy as np
import torch
from typing import List, Set
from urdf_parser_py.urdf import URDF
from mppi_solver.src.robot.urdfFks.transformation_matrix import *

from rclpy.logging import get_logger

class URDFparser(object):
    """Class that turns a chain from URDF to casadi functions."""
    actuated_types = ["prismatic", "revolute", "continuous"]

    def __init__(self, root_link: str = "base_link", end_links: List[str] = None):
        self._root_link = root_link
        if isinstance(end_links, str):
            self._end_links = [end_links]
        else:
            self._end_links = end_links if end_links else []

        self.robot_desc = None
        self._absolute_root_link = None
        self._active_joints = set()
        self._actuated_joints = list()
        self._joint_map = dict()
        self._degrees_of_freedom = 0
        self._link_names = list()
        self._joint_chain_list = None
        self._tf_fk = torch.eye(4)
        self._n_samples = 1
        self._n_timestep = 1
        self._n_mobile_dof = 0

        self.logger = get_logger("urdf_parser")
        self.fk_init = False
        self.init_transform_matrix = dict()
        self.init_axis = dict()

    def degrees_of_freedom(self):
        return self._degrees_of_freedom

    def joint_map(self):
        return self._joint_map
    
    def active_joints(self):
        return self._active_joints
    
    def link_names(self):
        return self._link_names

    def from_file(self, filename: str):
        self.robot_desc = URDF.from_xml_file(filename)
        self._extract_information()

    def _extract_information(self):
        self._detect_link_names()
        self._absolute_root_link = self.robot_desc.get_root()
        self._set_active_joints()
        self._set_actuated_joints()
        self._extract_degrees_of_freedom()
        self._set_joint_variable_map()

    def _set_active_joints(self):
        for end_lk in self._end_links:
            parent_link = end_lk
            while parent_link not in [self._root_link, self._absolute_root_link]:
                (parent_joint, parent_link) = self.robot_desc.parent_map[parent_link]
                self._active_joints.add(parent_joint)
                if parent_link == self._root_link:
                    break

    def _set_actuated_joints(self):
        self._actuated_joints = []
        for joint in self.robot_desc.joints:
            if joint.type in self.actuated_types:
                self._actuated_joints.append(joint.name)

    def _extract_degrees_of_freedom(self):
        self._degrees_of_freedom = 0
        for jn in self._active_joints:
            if jn in self._actuated_joints:
                self._degrees_of_freedom += 1

    def _set_joint_variable_map(self):
        self._joint_map = {}
        idx = 0
        for joint_name in self._actuated_joints:
            if joint_name in self._active_joints:
                self._joint_map[joint_name] = idx
                idx += 1

    def _is_active_joint(self, joint):
        parent_link = joint.parent
        while parent_link not in [self._root_link, self._absolute_root_link]:
            if parent_link in self._end_links:
                return False
            (parent_joint, parent_link) = self.robot_desc.parent_map[parent_link]
            if parent_joint in self._active_joints:
                return True
        if parent_link == self._root_link:
            return True
        return False

    def _detect_link_names(self):
        self._link_names = []
        for link in self.robot_desc.links:
            if link.name in self.robot_desc.parent_map:
                self._link_names.append(link.name)
        return self._link_names

    def _get_joint_chain(self, tip: str):
        if self.robot_desc is None:
            raise ValueError("Robot description not loaded.")
        chain = self.robot_desc.get_chain(self._absolute_root_link, tip)
        joint_list = []
        for item in chain:
            if item in self.robot_desc.joint_map:
                jnt = self.robot_desc.joint_map[item]
                if jnt.name in self._active_joints:
                    joint_list.append(jnt)
        return joint_list
    
    def _init_xyzrpy_tf_matrix(self):
        if self.robot_desc is None:
            raise ValueError("Robot description not loaded.")
        for jt in self._joint_chain_list:
            xyz = torch.tensor(jt.origin.xyz)
            rpy = torch.tensor(jt.origin.rpy)
            self.init_transform_matrix[jt.name] = make_transform_matrix(xyz, rpy)
        return
    
    def _init_axis(self):
        if self.robot_desc is None:
            raise ValueError("Robot description not loaded.")
        for jt in self._joint_chain_list:
            if jt.axis is None:
                axis = torch.tensor([1.0, 0.0, 0.0])
            else:
                axis = torch.tensor(jt.axis)
                if torch.linalg.norm(axis) < 1e-12:
                    axis = tensor([1.0, 0.0, 0.0])
                else:
                    axis = axis / torch.linalg.norm(axis)
            self.init_axis[jt.name] = axis
        return

    
    def forward_kinematics(self, q: torch.Tensor, free_floating: bool = False, base_move : bool = False):
        if self.robot_desc is None:
            raise ValueError("Robot description not loaded.")

        tf_fk = self._tf_fk.expand(self._n_samples, self._n_timestep, 4, 4).to(device=q.device).clone()
        tf_list = []

        if free_floating and base_move:
            tf_base = transformation_matrix_from_xyzrpy(q=q[:,:,:self._n_mobile_dof])
            tf_fk = tf_fk @ tf_base
            q = q[:,:,self._n_mobile_dof:]

        for jt in self._joint_chain_list:
            jtype = jt.type

            if jt.name in self._joint_map:
                q_idx = self._joint_map[jt.name]
                q_val = q[:,:,q_idx]
            else:
                q_val = torch.zeros([self._n_samples, self._n_timestep])

            if jtype == "fixed":
                tf_fk = torch.einsum('shij,jk->shik', tf_fk, self.init_transform_matrix[jt.name].to(q.device))
            elif jtype == "prismatic":
                tf_local = prismatic_transform(self.init_transform_matrix[jt.name].to(q.device), self.init_axis[jt.name], q_val)
                tf_fk = torch.einsum('shij,shjk->shik', tf_fk, tf_local)
            elif jtype in ["revolute", "continuous"]:
                tf_local = revolute_transform(self.init_transform_matrix[jt.name].to(q.device), self.init_axis[jt.name], q_val)
                tf_fk = torch.einsum('shij,shjk->shik', tf_fk, tf_local)
            else:
                tf_fk = torch.einsum('shij,jk->shik', tf_fk, self.init_transform_matrix[jt.name].to(q.device))
            tf_list.append(tf_fk)

        return tf_fk, tf_list


    def forward_kinematics_cpu(self, q: torch.Tensor, free_floating: bool = False, base_move : bool = False):
        if self.robot_desc is None:
            raise ValueError("Robot description not loaded.") 

        tf_fk = self._tf_fk.clone()
        tf_list = []

        if free_floating and base_move:
            tf_base = transformation_matrix_from_xyzrpy_cpu(q=q[:self._n_mobile_dof])
            tf_fk = tf_fk @ tf_base
            q = q[self._n_mobile_dof:]

        for jt in self._joint_chain_list:
            jtype = jt.type

            xyz = torch.tensor(jt.origin.xyz)
            rpy = torch.tensor(jt.origin.rpy)

            if jt.axis is None:
                axis = torch.tensor([1.0, 0.0, 0.0])
            else:
                axis = torch.tensor(jt.axis)

            if jt.name in self._joint_map:
                q_idx = self._joint_map[jt.name]
                q_val = q[q_idx]
            else:
                q_val = torch.zeros(1)

            if jtype == "fixed":
                tf_fk = tf_fk @ self.init_transform_matrix[jt.name]
            elif jtype == "prismatic":
                tf_local = prismatic_transform_cpu(self.init_transform_matrix[jt.name], axis, q_val)
                tf_fk = tf_fk @ tf_local
            elif jtype in ["revolute", "continuous"]:
                tf_local = revolute_transform_cpu(self.init_transform_matrix[jt.name], axis, q_val)
                tf_fk = tf_fk @ tf_local
            else:
                tf_fk = tf_fk @ self.init_transform_matrix[jt.name]
            tf_list.append(tf_fk)
        return tf_fk, tf_list
