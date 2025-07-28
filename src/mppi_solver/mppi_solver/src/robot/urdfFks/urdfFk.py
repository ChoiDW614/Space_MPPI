from typing import Union, List
import numpy as np
import torch

import mppi_solver.src.robot.urdfFks.urdfparser as u2c

from rclpy.logging import get_logger


class LinkNotInURDFError(Exception):
    pass

class URDFForwardKinematics():
    def __init__(self, urdf: str, root_link: str, end_links: List[str], base_type: str = "holonomic"):
        self.logger = get_logger("urdf_parser")

        self._urdf = urdf
        self._root_link = root_link
        self._end_links = end_links
        self._base_type = base_type

        self.robot = u2c.URDFparser(root_link, end_links)
        self.robot.from_file(urdf)

        self.robot._joint_chain_list = self.robot._get_joint_chain(self._end_links)
        self.robot._init_xyzrpy_tf_matrix()
        self.robot._init_axis()
        self._n_dof = self.robot.degrees_of_freedom()
        self._mount_transformation = torch.eye(4)
        self._mount_transformation_cpu = torch.eye(4)

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
        self.com_local_list = torch.stack(self.com_local_list, dim=-1).unsqueeze(0).unsqueeze(0).expand(-1, self.robot._n_timestep, -1, -1, -1) 


    def set_mount_transformation_deivce(self, device):
        self._mount_transformation = self._mount_transformation.to(device)


    def set_mount_transformation(self, mount_transformation):
        self._mount_transformation = mount_transformation
        self._mount_transformation_cpu = mount_transformation.cpu()


    def set_samples_and_timesteps(self, n_samples, n_timesteps, n_mobile_dof = 0):
        self.robot._n_samples = n_samples
        self.robot._n_timestep = n_timesteps
        self.robot._n_mobile_dof = n_mobile_dof
    

    def forward_kinematics(self,
        q: torch.Tensor,
        child_link: str,
        parent_link: Union[str, None] = None,
        init_transformation: torch.Tensor = None,
        free_floating: bool = False,
        base_move : bool = False
    ) -> torch.Tensor:
        import time


        if init_transformation is None:
            init_transformation = torch.eye(4, device=q.device)
        
        if parent_link is None:
            parent_link = self._root_link

        if child_link not in self.robot.link_names() and child_link != self._root_link:
            raise LinkNotInURDFError(f"The link {child_link} is not in the URDF. Valid links: {self.robot.link_names()}")
        if parent_link not in self.robot.link_names() and parent_link != self._root_link:
            raise LinkNotInURDFError( f"The link {parent_link} is not in the URDF. Valid links: {self.robot.link_names()}")

        if parent_link == self._root_link:
            tf_parent = torch.eye(4, device=q.device).expand(self.robot._n_samples, self.robot._n_timestep, 4, 4).clone()
        else:
            tf_parent, tf_list = self.robot.forward_kinematics(q, free_floating, base_move)
            tf_parent = torch.einsum('mn,np,btpj->btmj', init_transformation, self._mount_transformation, tf_parent)

        if child_link == self._root_link:
            tf_child = torch.eye(4, device=q.device).expand(self.robot._n_samples, self.robot._n_timestep, 4, 4).clone()
        else:
            tf_child, tf_list = self.robot.forward_kinematics(q, free_floating, base_move)
            tf_child = torch.einsum('mn,np,btpj->btmj', init_transformation, self._mount_transformation, tf_child)

        # self.logger.info(f"Parent TF {tf_parent}")
        # tf_paret = torch.eye
        
        tf_batch = torch.stack(tf_list, dim=-1)
        tf_list = torch.einsum('ij,ntjkl->ntikl', init_transformation @ self._mount_transformation, tf_batch)

        self.com_local_list = self.com_local_list.to(device=q.device)
        com_list = torch.einsum('ntijc,ntjlc->ntilc', tf_list[..., :7], self.com_local_list)[:,:,:3,3]

        tf_parent_inv = self.inverse_tfmatrix_batch(tf_parent)
        tf_parent_child = tf_parent_inv @ tf_child

        return tf_parent_child, tf_list, com_list


    def forward_kinematics_cpu(self,
        q: torch.Tensor,
        child_link: str,
        parent_link: Union[str, None] = None,
        init_transformation: torch.Tensor = None,
        free_floating: bool = False,
        base_move : bool = False
    ) -> torch.Tensor:

        if init_transformation is None:
            init_transformation = torch.eye(4)
        
        if parent_link is None:
            parent_link = self._root_link

        if child_link not in self.robot.link_names() and child_link != self._root_link:
            raise LinkNotInURDFError(f"The link {child_link} is not in the URDF. Valid links: {self.robot.link_names()}")
        if parent_link not in self.robot.link_names() and parent_link != self._root_link:
            raise LinkNotInURDFError( f"The link {parent_link} is not in the URDF. Valid links: {self.robot.link_names()}")

        if parent_link == self._root_link:
            tf_parent = torch.eye(4).clone()
        else:
            tf_parent, tf_list = self.robot.forward_kinematics_cpu(q, free_floating, base_move)
            tf_parent = init_transformation @ self._mount_transformation_cpu @ tf_parent

        if child_link == self._root_link:
            tf_child = torch.eye(4).clone()
        else:
            tf_child, tf_list = self.robot.forward_kinematics_cpu(q, free_floating, base_move)
            tf_child = init_transformation @ self._mount_transformation_cpu @ tf_child

        tf_batch = torch.stack(tf_list, dim=-1)
        tf_list = torch.einsum('ij,jkl->ikl', init_transformation @ self._mount_transformation_cpu, tf_batch)
        
        tf_parent_inv = torch.linalg.inv(tf_parent)
        tf_parent_child = tf_parent_inv @ tf_child

        return tf_parent_child, tf_list
    

    def inverse_tfmatrix_batch(self, tf_batch: torch.Tensor) -> torch.Tensor:
        R = tf_batch[..., :3,:3]
        t = tf_batch[..., :3, 3]

        R_T = R.transpose(-1, -2)

        t_new = -torch.matmul(R_T, t.unsqueeze(-1)).squeeze(-1)

        inv_tf = tf_batch.clone()
        inv_tf[..., :3, :3] = R_T
        inv_tf[..., :3, 3]  = t_new
        return inv_tf
    