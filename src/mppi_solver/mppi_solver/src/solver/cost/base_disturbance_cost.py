import torch
from rclpy.logging import get_logger

from mppi_solver.src.utils.pose import Pose
from mppi_solver.src.robot.urdfFks.transformation_matrix import *
from mppi_solver.src.utils.exp_coordi import *
from mppi_solver.src.utils.canadarm_config import CanadarmConfiguration

class BaseDisturbanceCost:
    def __init__(self, n_action, device):
        self.logger = get_logger("Base_Disturbance_Cost")
        self.device = device
        self.n_action = n_action

        self.base_pos = torch.zeros(3, device=self.device)
        self.base_lin_vel = torch.zeros(3, device=self.device)
        self.base_ang_vel = torch.zeros(3, device=self.device)

        self.joint_list = torch.zeros((self.n_action+1, 4, 4), device=self.device)
        self.joint_list2 = torch.zeros((self.n_action+1, 4, 4), device=self.device)
        self.com_list = torch.zeros((self.n_action, 4, 4), device=self.device)

        # Canadarm Configuration from URDF
        self.canadarm_config = CanadarmConfiguration(self.n_action, self.device)
        
        self.p_list = torch.zeros((self.n_action, 3), device=self.device)
        self.K_list = torch.zeros((self.n_action, 3), device=self.device)

        self.theta_dot = torch.ones(self.n_action)*0.1

        # calculation variables
        self.tmp = torch.zeros((self.n_action, 3), device=self.device)
        self.tmp2 = torch.zeros((self.n_action, 3), device=self.device)
        self.js = torch.zeros((6, 7), device=self.device)
        self.js_auto = torch.zeros((6, 7), device=self.device)
        self.js_auto2 = torch.zeros((6, 7), device=self.device)
        return


    def compute_base_disturbance_cost(self, base_pos: Pose, q: torch.tensor, base_lin_vel: torch.Tensor, base_ang_vel: torch.Tensor):
        self.base_pos = base_pos
        self.base_lin_vel = base_lin_vel
        self.base_ang_vel = base_ang_vel

        self.base_pos = base_pos.pose.to(self.device)
        self.base_ori = base_pos.tf_matrix(self.device)[0:3, 0:3]
        self.base_lin_vel = torch.zeros(3, device=self.device)
        self.base_ang_vel = torch.zeros(3, device=self.device)

        # Compute positions
        # self.compute_joint_positions(self.base_pos, self.base_ori, q)
        self.compute_fk_skew(self.base_pos, self.base_ori, q)
        js = self.compute_jacobian_space(self.base_pos, self.base_ori, q)
        # self.compute_link_positions(self.base_pos, self.base_ori, q)
        
        # self.logger.info(f"joint: {self.joint_origins[-1,:3,3]}")

        # # Compute velocities
        # v_list, w_list = self.compute_link_velocities(self.base_pos, self.base_lin_vel, self.base_ang_vel, self.theta_dot, self.K_list, self.p_list, link_positions=r_list)
        # self.logger.info(f"Link linear velocities: {v_list}")
        # self.logger.info(f"Link angular velocities: {w_list}")

        # # End-effector kinematics
        # r_e = r_list[-1]
        # J_b, J_m = self.compute_jacobians(self.base_pos, r_e, self.p_list)
        # self.logger.info(f"J_b shape: {J_b.shape}, J_m shape: {J_m.shape}")


        return
    

    def compute_prev_base_disturbance_cost(self, uSample):


        return 
    

    def compute_joint_positions(self, base_pos: torch.Tensor, base_ori: torch.Tensor, q: torch.Tensor):
        T = torch.eye(4, device=self.device)
        T[:3, 3] = base_pos
        T[:3, :3] = base_ori
        T = T @ self.canadarm_config.T_to_base

        for i in range(self.n_action):
            T_origin = make_transform_matrix(self.canadarm_config.rpyxyz[i,3:6], self.canadarm_config.rpyxyz[i,0:3]).to(self.device)

            R_i = axis_angle_to_matrix(self.canadarm_config.axis[i,:], q[i])
            T_rot = torch.eye(4, device=self.device)
            T_rot[:3,:3] = R_i

            T = T @ T_origin @ T_rot
            self.joint_list[i,:,:] = T.clone()

        T_tip = make_transform_matrix(self.canadarm_config.rpyxyz_tip[3:6], self.canadarm_config.rpyxyz_tip[0:3]).to(self.device)
        T = T @ T_tip
        self.joint_list[-1,:,:] = T.clone()
        return
    

    def compute_fk_skew(self, base_pos: torch.Tensor, base_ori: torch.Tensor, q: torch.Tensor):
        with torch.no_grad():
            T_base = torch.eye(4, device=self.device)
            T_base[:3,3] = base_pos
            T_base[:3,:3] = base_ori
            T_base = T_base @ self.canadarm_config.T_to_base

            T_exp = torch.eye(4, device=self.device)
            for i in range(self.n_action):
                s_i = self.canadarm_config.s_list[:,i].T
                s_skew = vec6_to_skew4(s_i * q[i])
                mat_exp = skew4_to_matrix_exp4(s_skew)
                T_exp = T_exp @ mat_exp
                self.joint_list2[i,:,:] = (T_base @ T_exp @ self.canadarm_config.m_list[i,:,:]).clone()
        return


    def compute_jacobian_space(self, base_pos: torch.Tensor, base_ori: torch.Tensor, q: torch.Tensor):
        Js = self.canadarm_config.s_list.clone()

        with torch.no_grad():
            T_base = torch.eye(4, device=self.device)
            T_base[:3,3] = base_pos
            T_base[:3,:3] = base_ori
            T_base = T_base @ self.canadarm_config.T_to_base

            T_exp = torch.eye(4, device=self.device)
            for i in range(1, self.n_action):
                s_i = self.canadarm_config.s_list[:,i-1].T
                s_skew = vec6_to_skew4(s_i * q[i-1])
                mat_exp = skew4_to_matrix_exp4(s_skew)
                T_exp = T_exp @ mat_exp

                adj_T = htm_adj(T_exp)
                J_col = adj_T @ self.canadarm_config.s_list[:,i]
                Js[:,i] = J_col
                self.js = Js
        return Js




    def pose_to_transform(self, rpyxyz: torch.Tensor) -> torch.Tensor:
        cr, sr = torch.cos(rpyxyz[0]), torch.sin(rpyxyz[0])
        cp, sp = torch.cos(rpyxyz[1]), torch.sin(rpyxyz[1])
        cy, sy = torch.cos(rpyxyz[2]), torch.sin(rpyxyz[2])

        T = torch.eye(4, device=self.device)
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


def axis_angle_to_quaternion(axis, angle):
    axis = torch.asarray(axis, dtype=torch.float64)
    axis = axis / torch.norm(axis)
    eta = torch.cos(angle / 2.0)
    q_vec = axis * torch.sin(angle / 2.0)
    return torch.concatenate(([eta], q_vec))


def quaternion_derivative(quaternion, angular_velocity):
    eta = quaternion[0]
    q = quaternion[1:]
    w = torch.asarray(angular_velocity, dtype=torch.float64)

    dot_eta = -0.5 * torch.dot(q, w)

    q_cross = torch.tensor([[0,     -q[2],  q[1]],
                            [q[2],  0,     -q[0]],
                            [-q[1], q[0],  0   ]])
    dot_q = 0.5 * (eta * torch.eye(3) + q_cross).dot(w)
    dot_q = 0.5 * torch.dot(eta * torch.eye(3) + q_cross, w)
    return torch.concatenate(([dot_eta], dot_q))


def quaternion_error(current_q, desired_q):
    eta_b, q_b = current_q[0], current_q[1:]
    eta_d, q_d = desired_q[0], desired_q[1:]

    eta_err = eta_b * eta_d - torch.dot(q_b, q_d)
    q_err = eta_b * q_d + eta_d * q_b + torch.cross(q_b, q_d)
    return torch.concatenate(([eta_err], q_err))
