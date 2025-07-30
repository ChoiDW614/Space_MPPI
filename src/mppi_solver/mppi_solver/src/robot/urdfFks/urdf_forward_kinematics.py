import numpy as np
import torch
import torch.nn as nn
from typing import List, Set, Union
from urdf_parser_py.urdf import URDF
from mppi_solver.src.robot.urdfFks.transformation_matrix import *

from rclpy.logging import get_logger


class LinkNotInURDFError(Exception):
    pass


class URDFForwardKinematics(nn.Module):
    actuated_types = ["prismatic", "revolute", "continuous"]
    def __init__(self, params, urdf: str, root_link: str = "base_link", end_links: List[str] = None, base_type: str = "holonomic", tensor_args: dict=None):
        super().__init__()
        self.logger = get_logger('URDF_FK')

        self.tensor_args = tensor_args
        self.n_samples = params['mppi']['sample']
        self.n_horizon = params['mppi']['horizon']
        self.is_free_floating = params['mppi']['free_floating']
        self.is_base_move = params['mppi']['base_move']
        self.n_mobile_dof = 0

        self._urdf = urdf
        self._root_link = root_link
        self._end_links = end_links
        self._base_type = base_type

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
        self.init_transform_matrix = dict()
        self.init_axis = dict()

        self.from_file(self._urdf)

        self._joint_chain_list = self._get_joint_chain(end_links)
        self._init_xyzrpy_tf_matrix(self.tensor_args)
        self._init_axis(self.tensor_args)
        self.n_dof = self.degrees_of_freedom()
        self._mount_transformation_cpu = torch.tensor([[1, 0, 0,   0],
                                                       [0,-1, 0,   0],
                                                       [0, 0,-1, 3.6],
                                                       [0, 0, 0,   1]], dtype=self.tensor_args['dtype'])
        self._mount_transformation = self._mount_transformation_cpu.to(**self.tensor_args)
        self.n_joint_chain_list = len(self._joint_chain_list)
        self._tf_fk = torch.eye(4, **self.tensor_args)
        self._tf_list = torch.zeros((4, 4, self.n_joint_chain_list), **self.tensor_args)

        # Compute COM of Each Link
        com_local_1 = torch.tensor([[0,0,-1,0.25082],
                                    [0,1,0,0],
                                    [1,0,0,-0.175],
                                    [0,0,0,1]])
        com_local_2 = torch.tensor([[1,0,0,0.175],
                                    [0,1,0,0],
                                    [0,0,1,-0.25082],
                                    [0,0,0,1]])
        com_local_3 = torch.tensor([[1,0,0,4.0],
                                    [0,-1,0,0],
                                    [0,0,-1,-0.175],
                                    [0,0,0,1]])
        com_local_4 = torch.tensor([[1,0,0,-3.6],
                                    [0,1,0,0],
                                    [0,0,1,-0.175],
                                    [0,0,0,1]])
        com_local_5 = torch.tensor([[1,0,0, 0],
                                    [0,0,-1,0],
                                    [0,1,0,0],
                                    [0,0,0,1]])
        com_local_6 = torch.tensor([[0,0,-1, 0],
                                    [0,1,0,0],
                                    [1,0,0,0],
                                    [0,0,0,1]])
        com_local_7 = torch.tensor([[0,0,1, 0],
                                    [0,1,0,0],
                                    [-1,0,0,-0.5],
                                    [0,0,0,1]])
        self.com_local_list = [com_local_1, com_local_2, com_local_3, com_local_4, com_local_5, com_local_6, com_local_7]
        self.com_local_list = torch.stack(self.com_local_list, dim=-1).unsqueeze(0).unsqueeze(0).expand(-1, self.n_horizon, -1, -1, -1) 
        self.com_local_list_gpu = self.com_local_list.to(**self.tensor_args)
        return
    

    def forward(self, q: torch.Tensor, child_link: str, parent_link: Union[str, None] = None,
                init_transformation: torch.Tensor = None) -> torch.Tensor:
        with torch.no_grad():
            if init_transformation is None:
                init_transformation = torch.eye(4, **self.tensor_args)
            else:
                init_transformation = init_transformation.to(**self.tensor_args).clone()

            if parent_link is None:
                parent_link = self._root_link

            if child_link not in self.link_names() and child_link != self._root_link:
                raise LinkNotInURDFError(f"The link {child_link} is not in the URDF. Valid links: {self.link_names()}")
            if parent_link not in self.link_names() and parent_link != self._root_link:
                raise LinkNotInURDFError( f"The link {parent_link} is not in the URDF. Valid links: {self.link_names()}")


            if parent_link == self._root_link:
                tf_parent = torch.eye(4, **self.tensor_args).expand(self.n_samples, self.n_horizon, 4, 4).clone()
            else:
                tf_parent, tf_list = self._compute_chain_fk(q)
                tf_parent = torch.einsum('mn,np,btpj->btmj', init_transformation, self._mount_transformation, tf_parent)

            if child_link == self._root_link:
                tf_child = torch.eye(4, **self.tensor_args).expand(self.n_samples, self.n_horizon, 4, 4).clone()
            else:
                tf_child, tf_list = self._compute_chain_fk(q)
                tf_child = torch.einsum('mn,np,btpj->btmj', init_transformation, self._mount_transformation, tf_child)

            tf_list = torch.einsum('ip,pj,ntjkl->ntikl', init_transformation, self._mount_transformation, tf_list)

            tf_list_rot = tf_list[...,:3,:3,:7]
            tf_list_trans = tf_list[...,:3, 3,:7]
            tf_list_rot = torch.sum(tf_list_rot * self.com_local_list_gpu[...,:3,3,:].unsqueeze(2), dim=3)
            com_list = tf_list_rot + tf_list_trans

            tf_parent_inv = self.inverse_tfmatrix_batch(tf_parent)
            tf_parent_child = torch.einsum('ntij,ntjk->ntik',tf_parent_inv, tf_child)

        return tf_parent_child, tf_list, com_list


    def _compute_chain_fk(self, q: torch.Tensor):
        if q.dim() == 1:
            q = q.unsqueeze(0).unsqueeze(0)

        tf_fk = self._tf_fk.expand(self.n_samples, self.n_horizon, 4, 4).clone()
        tf_list = self._tf_list.expand(self.n_samples, self.n_horizon, 4, 4, self.n_joint_chain_list).clone()

        if self.is_free_floating and self.is_base_move:
            tf_base = transformation_matrix_from_xyzrpy(q=q[...,:self.n_mobile_dof])
            tf_fk = tf_fk @ tf_base
            q = q[...,self.n_mobile_dof:]

        iter = 1
        for jt in self._joint_chain_list:
            jtype = jt.type

            if jt.name in self._joint_map:
                q_idx = self._joint_map[jt.name]
                q_val = q[...,q_idx]
            else:
                q_val = torch.zeros([self.n_samples, self.n_horizon])

            if jtype == "fixed":
                tf_fk = torch.einsum('shij,jk->shik', tf_fk, self.init_transform_matrix[jt.name])
            elif jtype == "prismatic":
                tf_local = prismatic_transform(self.init_transform_matrix[jt.name], self.init_axis[jt.name], q_val)
                tf_fk = torch.einsum('shij,shjk->shik', tf_fk, tf_local)
            elif jtype in ["revolute", "continuous"]:
                tf_local = revolute_transform(self.init_transform_matrix[jt.name], self.init_axis[jt.name], q_val)
                tf_fk = torch.einsum('shij,shjk->shik', tf_fk, tf_local)
            else:
                tf_fk = torch.einsum('shij,jk->shik', tf_fk, self.init_transform_matrix[jt.name])

            tf_list[:,:,:,:, self._joint_chain_list.index(jt)] = tf_fk.clone()
            iter += 1
        return tf_fk, tf_list


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
    
    def _init_xyzrpy_tf_matrix(self, tensor_args):
        if self.robot_desc is None:
            raise ValueError("Robot description not loaded.")
        for jt in self._joint_chain_list:
            xyz = torch.tensor(jt.origin.xyz)
            rpy = torch.tensor(jt.origin.rpy)
            self.init_transform_matrix[jt.name] = make_transform_matrix(xyz, rpy).to(**tensor_args)
        return
    
    def _init_axis(self, tensor_args):
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
            self.init_axis[jt.name] = axis.to(**tensor_args)
        return

    def set_mount_transformation(self, mount_transformation: torch.Tensor):
        self._mount_transformation = mount_transformation
        self._mount_transformation_cpu = mount_transformation.cpu()

    def set_samples_and_timesteps(self, n_samples: int, n_horizon: int, n_mobile_dof: int = 0):
        self.n_samples = n_samples
        self.n_horizon = n_horizon
        self.n_mobile_dof = n_mobile_dof

    def inverse_tfmatrix_batch(self, tf_batch: torch.Tensor) -> torch.Tensor:
        R = tf_batch[..., :3,:3]
        t = tf_batch[..., :3, 3]

        R_T = R.transpose(-1, -2)

        t_new = -torch.matmul(R_T, t.unsqueeze(-1)).squeeze(-1)

        inv_tf = tf_batch.clone()
        inv_tf[..., :3,:3] = R_T
        inv_tf[..., :3, 3] = t_new
        return inv_tf
    