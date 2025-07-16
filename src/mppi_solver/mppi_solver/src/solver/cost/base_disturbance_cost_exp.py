import torch
from rclpy.logging import get_logger

from mppi_solver.src.utils.pose import Pose
from mppi_solver.src.robot.urdfFks.transformation_matrix import *
from mppi_solver.src.utils.exp_coordi import *
from mppi_solver.src.utils.canadarm_config import CanadarmConfiguration

class BaseDisturbanceCost:
    def __init__(self, n_action, tensor_args):
        self.logger = get_logger("Base_Disturbance_Cost")
        self.tensor_args = tensor_args
        self.n_action = n_action

        self.base_pos = torch.zeros(3, **self.tensor_args)
        self.base_lin_vel = torch.zeros(3, **self.tensor_args)
        self.base_ang_vel = torch.zeros(3, **self.tensor_args)

        self.joint_list = torch.zeros((self.n_action+1, 4, 4), **self.tensor_args)
        self.joint_list2 = torch.zeros((self.n_action+1, 4, 4), **self.tensor_args)
        self.com_list = torch.zeros((self.n_action, 4, 4), **self.tensor_args)

        # Canadarm Configuration from URDF
        self.canadarm_config = CanadarmConfiguration(self.n_action, self.tensor_args)
        
        self.p_list = torch.zeros((self.n_action, 3), **self.tensor_args)
        self.K_list = torch.zeros((self.n_action, 3), **self.tensor_args)

        self.theta_dot = torch.ones(self.n_action)*0.1

        # calculation variables
        self.tmp = torch.zeros((self.n_action, 3), **self.tensor_args)
        self.tmp2 = torch.zeros((self.n_action, 3), **self.tensor_args)
        self.js = torch.zeros((6, 7), **self.tensor_args)
        self.js_auto = torch.zeros((6, 7), **self.tensor_args)
        self.js_auto2 = torch.zeros((6, 7), **self.tensor_args)
        return


    def compute_base_disturbance_cost(self, base_pos: Pose, q: torch.tensor, base_lin_vel: torch.Tensor, base_ang_vel: torch.Tensor):
        self.base_pos = base_pos
        self.base_lin_vel = base_lin_vel
        self.base_ang_vel = base_ang_vel

        self.base_pos = base_pos.pose.to(self.device)
        self.base_ori = base_pos.tf_matrix(self.device)[0:3, 0:3]
        self.base_lin_vel = torch.zeros(3, **self.tensor_args)
        self.base_ang_vel = torch.zeros(3, **self.tensor_args)

        # Compute positions
        # self.compute_joint_positions(self.base_pos, self.base_ori, q)
        self.compute_joint_positions(self.base_pos, self.base_ori, q)
        # self.compute_link_positions(self.base_pos, self.base_ori, q)
        
        return
    

    def compute_prev_base_disturbance_cost(self, uSample):
        return 
    

    # def compute_joint_positions(self, base_pos: torch.Tensor, base_ori: torch.Tensor, q: torch.Tensor):
    #     T = torch.eye(4, **self.tensor_args)
    #     T[:3, 3] = base_pos
    #     T[:3, :3] = base_ori
    #     T = T @ self.canadarm_config.T_to_base

    #     for i in range(self.n_action):
    #         T_origin = make_transform_matrix(self.canadarm_config.rpyxyz[i,3:6], self.canadarm_config.rpyxyz[i,0:3]).to(self.device)

    #         R_i = axis_angle_to_matrix(self.canadarm_config.axis[i,:], q[i])
    #         T_rot = torch.eye(4, **self.tensor_args)
    #         T_rot[:3,:3] = R_i

    #         T = T @ T_origin @ T_rot
    #         self.joint_list[i,:,:] = T.clone()

    #     T_tip = make_transform_matrix(self.canadarm_config.rpyxyz_tip[3:6], self.canadarm_config.rpyxyz_tip[0:3]).to(self.device)
    #     T = T @ T_tip
    #     self.joint_list[-1,:,:] = T.clone()
    #     return
    

    def compute_joint_positions(self, base_pos: torch.Tensor, base_ori: torch.Tensor, q: torch.Tensor):
        with torch.no_grad():
            T_base = torch.eye(4, **self.tensor_args)
            T_base[:3,3] = base_pos
            T_base[:3,:3] = base_ori
            T_base = T_base @ self.canadarm_config.T_to_base

            T_exp = torch.eye(4, **self.tensor_args)
            for i in range(self.n_action):
                s_i = self.canadarm_config.s_list[:,i].T
                s_skew = vec6_to_skew4(s_i * q[i])
                mat_exp = skew4_to_matrix_exp4(s_skew)
                T_exp = T_exp @ mat_exp
                self.joint_list[i,:,:] = (T_base @ T_exp @ self.canadarm_config.m_list[i,:,:]).clone()
        return
    
    def compute_link_positions(self, base_pos: torch.Tensor, base_ori: torch.Tensor, q: torch.Tensor):
        with torch.no_grad():
            T_base = torch.eye(4, **self.tensor_args)
            T_base[:3,3] = base_pos
            T_base[:3,:3] = base_ori
            T_base = T_base @ self.canadarm_config.T_to_base

            T_exp = torch.eye(4, **self.tensor_args)
            for i in range(self.n_action):
                s_i = self.canadarm_config.s_list[:,i].T
                s_skew = vec6_to_skew4(s_i * q[i])
                mat_exp = skew4_to_matrix_exp4(s_skew)
                T_exp = T_exp @ mat_exp
                self.joint_list[i,:,:] = (T_base @ T_exp @ self.canadarm_config.m_list[i,:,:]).clone()
        return


    # def compute_jacobian_space(self, base_pos: torch.Tensor, base_ori: torch.Tensor, q: torch.Tensor):
    #     Js = self.canadarm_config.s_list.clone()

    #     with torch.no_grad():
    #         T_base = torch.eye(4, **self.tensor_args)
    #         T_base[:3,3] = base_pos
    #         T_base[:3,:3] = base_ori
    #         T_base = T_base @ self.canadarm_config.T_to_base

    #         T_exp = torch.eye(4, **self.tensor_args)
    #         for i in range(1, self.n_action):
    #             s_i = self.canadarm_config.s_list[:,i-1].T
    #             s_skew = vec6_to_skew4(s_i * q[i-1])
    #             mat_exp = skew4_to_matrix_exp4(s_skew)
    #             T_exp = T_exp @ mat_exp

    #             adj_T = htm_adj(T_exp)
    #             J_col = adj_T @ self.canadarm_config.s_list[:,i]
    #             Js[:,i] = J_col
    #             self.js = Js
    #     return Js


    def compute_jacobian_space(self, link_pose_list):
        eef = link_pose_list[-1].clone()
        n_samples, n_timesteps, _, _ = eef.shape
        jacob = torch.zeros((n_samples, n_timesteps, 6, 7), **self.tensor_args)
        p_eef = eef[:,:,:3,3]

        for i in range(7):
            p = link_pose_list[i][:,:,:3, 3].clone()
            del_p = p_eef - p
            rot = link_pose_list[i][:,:,:3,self.axis_list[i]].clone()
            if i == 4:
                rot *= -1

            jacob[:,:,:3,i] = torch.cross(rot, del_p)
            jacob[:,:,3:,i] = rot.clone()
        return jacob


    def pose_to_transform(self, rpyxyz: torch.Tensor) -> torch.Tensor:
        cr, sr = torch.cos(rpyxyz[0]), torch.sin(rpyxyz[0])
        cp, sp = torch.cos(rpyxyz[1]), torch.sin(rpyxyz[1])
        cy, sy = torch.cos(rpyxyz[2]), torch.sin(rpyxyz[2])

        T = torch.eye(4, **self.tensor_args)
        T[:3, :3] = torch.stack([torch.stack([cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr]),
                                 torch.stack([sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr]),
                                 torch.stack([-sp, cp*sr, cp*cr])], dim=0)
        return T



def axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    ux, uy, uz = axis
    c = torch.cos(angle)
    s = torch.sin(angle)
    one_c = 1 - c

    R = torch.stack([
        torch.stack([c + ux*ux*one_c,      ux*uy*one_c - uz*s, ux*uz*one_c + uy*s]),
        torch.stack([uy*ux*one_c + uz*s,   c + uy*uy*one_c,    uy*uz*one_c - ux*s]),
        torch.stack([uz*ux*one_c - uy*s,   uz*uy*one_c + ux*s, c + uz*uz*one_c   ])
    ])
    return R
