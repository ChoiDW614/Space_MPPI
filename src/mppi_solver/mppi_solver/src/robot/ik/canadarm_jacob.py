import torch
import torch.nn as nn
import time
from rclpy.logging import get_logger


class CanadarmJacob(nn.Module):
    def __init__(self, params, tensor_args):
        super().__init__()
        self.logger = get_logger("Canadarm_Jacobian")
        self.tensor_args = tensor_args

        self.n_samples = params['mppi']['sample']
        self.n_horizon = params['mppi']['horizon']
        self.n_action = params['mppi']['action']
        self.n_mani_dof = params['mppi']['manipulator_dof']

        self.axis_list = torch.tensor([2, 0, 2, 2, 2, 0, 2], dtype=torch.long, device=self.tensor_args['device'])
        self.link_list = torch.arange(self.n_action, device=self.tensor_args['device'])   

        self.mass_list = torch.tensor([105.98, 105.98, 314.98, 279.2, 105.98, 105.98, 243.66], **self.tensor_args)
        self.total_mass = torch.sum(self.mass_list) + 100000.0 + 243.66

        self.inertial_list = torch.zeros((7, 3, 3), **self.tensor_args)
        self.inertial_list[0,0,0] = 12.19
        self.inertial_list[0,1,1] = 12.19
        self.inertial_list[0,2,2] = 3.061

        self.inertial_list[1,0,0] = 12.19
        self.inertial_list[1,1,1] = 12.19
        self.inertial_list[1,2,2] = 3.061

        self.inertial_list[2,0,0] = 15.41
        self.inertial_list[2,1,1] = 2094.71
        self.inertial_list[2,2,2] = 2103.19

        self.inertial_list[3,0,0] = 9.522
        self.inertial_list[3,1,1] = 1966.28
        self.inertial_list[3,2,2] = 1966.28

        self.inertial_list[4,0,0] = 8.305
        self.inertial_list[4,1,1] = 3.061
        self.inertial_list[4,2,2] = 8.0386

        self.inertial_list[5,0,0] = 12.13
        self.inertial_list[5,1,1] = 12.13
        self.inertial_list[5,2,2] = 3.061

        self.inertial_list[6,0,0] = 9.336
        self.inertial_list[6,1,1] = 44.41
        self.inertial_list[6,2,2] = 44.41
        
        self.tf_link0 = torch.tensor([[1, 0,   0, 0],
                                     [0,-1,   0, 0],
                                     [0, 0, 1.3, 6],
                                     [0, 0,   0, 1]], **self.tensor_args)
            
        self.com_link0_local = torch.tensor([[1, 0, 0,   0],
                                             [0, 1, 0,   0],
                                             [0, 0, 1, 0.5],
                                             [0, 0, 0,   1]], **self.tensor_args)
        
        link0_com = torch.matmul(self.tf_link0, self.com_link0_local)
        self.base_com_pose = link0_com[:3,3].clone() * 243.66 / (100000.0 + 243.66)

        # for optimization 
        self.I_0 = torch.diag(torch.tensor([69585.02, 69585.02, 66666.664], **self.tensor_args))
        self.I_sum = torch.sum(self.inertial_list, dim=0)
        self.mass_list_sum = torch.sum(self.mass_list)
        self.lower_tri_mask = torch.tril(torch.ones(self.n_mani_dof, self.n_mani_dof, dtype=torch.bool, device=self.tensor_args['device'])) # lower triangular matrix
        self.J_tw_mass_mask = (self.lower_tri_mask * self.mass_list.view(7,1)).view(1, 1, self.n_mani_dof, self.n_mani_dof, 1)
        self.lower_tri_mask = self.lower_tri_mask.view(1, 1, 7, 7, 1)
        self.cross_eps = torch.tensor([[[0,0,0],[0,0,1],[0,-1,0]], [[0,0,-1],[0,0,0],[1,0,0]],[[0,1,0],[-1,0,0],[0,0,0]]], **self.tensor_args)


    def forward(self, com_list: torch.Tensor, link_pose_list: torch.Tensor, jabobian: torch.Tensor = None, bm: bool = False) -> torch.Tensor:
        if bm:
            return self.compute_jacob_bm(com_list, link_pose_list, jabobian)
        else:
            return self.compute_jacobian(link_pose_list)


    def compute_jacobian(self, link_pose_list: torch.Tensor):
        with torch.no_grad():
            n_samples, n_horizon, _, _, _ = link_pose_list.shape
            jacob = torch.zeros((n_samples, n_horizon, 6, self.n_action), **self.tensor_args)

            del_p = link_pose_list[:,:,:3,3,-2].unsqueeze(-1) - link_pose_list[:,:,:3,3,:self.n_action]
            rot = link_pose_list[:,:,:3, self.axis_list, self.link_list]
            rot[...,4] *= -1

            jacob[:,:,:3,:] = torch.cross(rot, del_p, dim=2)
            jacob[:,:,3:,:] = rot.clone()
        return jacob
    

    def compute_jacobian_cpu(self, link_pose_list: torch.Tensor):
        jacob = torch.zeros((6, self.n_action))

        del_p = link_pose_list[:3,3,-2].unsqueeze(-1) - link_pose_list[:3,3,:self.n_action]
        rot = link_pose_list[:3, self.axis_list.cpu(), self.link_list.cpu()]
        rot[...,4] *= -1

        jacob[3:, :] = rot
        jacob[:3, :] = torch.cross(rot, del_p, dim=0)
        return jacob
    

    def compute_base_jacob(self, base_pose, link_pose_list):
        eef = link_pose_list[-2].clone()
        n_samples, n_timesteps, _, _ = eef.shape
        eef_p = eef[:,:, :3, 3].clone()
        eef_rot = eef[:,:, :3, :3].clone()
        base_p = base_pose[:3, 3].clone()
        del_p = eef_p - base_p
        skew_p = self.make_skew_mat(del_p)

        device = eef.device
        base_jacob = torch.zeros((n_samples, n_timesteps, 6, 6), device=device)
        base_jacob[...,:,:] = torch.eye((6), device=device)
        base_jacob[...,:3,:] = base_jacob[...,:3,:] + torch.matmul(skew_p, base_jacob[...,3:,:])

        return base_jacob
    

    # 이게 일단은 안쓰이나봄?
    # compute jacobian considering floating base
    def compute_jacob_floating(self, base_pose, jacobian):
        adj_mat = self.compute_adj_mat(base_pose)
        adj_mat = adj_mat.to(jacobian.device)
        manip_jacob = jacobian.clone()

        return torch.matmul(adj_mat, manip_jacob)
    

    def compute_jacob_bm(self, com_list: torch.Tensor, link_pose_list: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
        n_samples, n_horizon, _, _ = com_list.shape
        # com_list:       [s, h, 3, 7]
        # link_pose_list: [s, h, 4, 4, 8]
        # jacobian:       [s, h, 6, 7]
        # J_ri:           [s, h, 3, 7]
        system_com = self.compute_system_com(com_list)
        r_og = system_com - self.base_com_pose.view(1, 1, -1)                           # [s, h, 3]
        r_og_skew = self.make_skew_mat(r_og).contiguous()                               # [s, h, 3, 3]

        J_ri = jacobian[:,:,:3,:]                                                       # [s, h, 3, 7]
        rp_i = com_list - link_pose_list[..., :3, 3, :self.n_mani_dof]                  # [s, h, 3, 7]

        J_ri = J_ri.permute(0,1,3,2).unsqueeze(2).expand(-1, -1, 7, -1, -1)
        rp_i = rp_i.permute(0,1,3,2).unsqueeze(3).expand(-1, -1, -1, 7, -1)

        cross_all = torch.cross(J_ri, rp_i, dim=-1)                                     # [s, h, 7, 7, 3]
        J_tw = torch.sum(cross_all * self.J_tw_mass_mask, dim=2).permute(0, 1, 3, 2).contiguous() # [s, h, 3, 7]

        r_og_skew_mul = self.make_skew_mul_mat(r_og)                                    # [s, h, 3, 3]
        H_w = self.I_sum.view(1, 1, 3, 3) + self.mass_list_sum * r_og_skew_mul + self.I_0

        r_oi = com_list - self.base_com_pose.view(1, 1, 3, 1)                           # [s, h, 3, 7]
        skew_r_oi = self.make_skew_mat(r_oi.permute(0,1,3,2))                           # [s, h, 7, 3, 3]

        masked_Jri   = J_ri * self.lower_tri_mask                                       # [s, h, 7, 7, 3]
        J_ti_mass    = cross_all * self.J_tw_mass_mask                                  # [s, h, 7, 7, 3]

        H_w_phi_1 = torch.einsum('iad,btijd->btaj', self.inertial_list, masked_Jri)
        H_w_phi_2 = torch.einsum('btiac,btijc->btaj', skew_r_oi, J_ti_mass)

        H_w_phi = H_w_phi_1 + H_w_phi_2

        H_theta = H_w_phi - torch.einsum('btij,btjk->btik', r_og_skew, J_tw)

        H_s = self.total_mass * torch.matmul(r_og_skew, r_og_skew) + H_w
        H_s_inv = self.invert_3x3(H_s)

        jacob_bm = torch.zeros((n_samples, n_horizon, 6, 7), **self.tensor_args)
        jacob_bm[:,:,:3,:] = -(J_tw / self.total_mass + torch.matmul(torch.matmul(r_og_skew, H_s_inv), H_theta))
        jacob_bm[:,:,3:,:] = - torch.matmul(H_s_inv, H_theta)

        return jacob_bm
    

    def compute_jacob_bm_v2(self, com_list, link_pose_list, jacobian):
        n_samples, n_timesteps, _, _ = com_list.shape

        J_tw = torch.zeros((n_samples, n_timesteps, 3, 7)).to(**self.tensor_args)
        J_ri = torch.zeros((n_samples, n_timesteps, 3, 7)).to(**self.tensor_args)
        H_w = torch.zeros((n_samples, n_timesteps, 3, 3)).to(**self.tensor_args)
        H_w_phi = torch.zeros((n_samples, n_timesteps, 3, 7)).to(**self.tensor_args)

        system_com = self.compute_system_com(com_list)
        r_og  = system_com - self.base_com_pose.view(1,1,-1)
        skew_r_og = self.make_skew_mat(r_og)

        for i in range(7):
            J_ti = torch.zeros((n_samples, n_timesteps, 3, 7)).to(**self.tensor_args)
            J_ri[:,:,:,:i+1] = jacobian[:,:,:3,:i+1]
            for j in range(i+1):
                J_ti[:,:,:,j] = torch.cross(J_ri[:,:,:,j], com_list[...,i].clone()-link_pose_list[:,:,:3,3,i].clone(), dim=-1)
            J_tw += J_ti.clone() * self.mass_list[i]
            r_oi = com_list[...,i].clone() - self.base_com_pose.clone()
            skew_r_oi = self.make_skew_mat(r_oi)
            H_w += self.inertial_list[i,:,:].view(1,1,3,3) + self.mass_list[i] * torch.matmul(skew_r_og.transpose(-1,2), skew_r_og)
            H_w_phi += torch.matmul(self.inertial_list[i,:,:].view(1,1,3,3), J_ri) + self.mass_list[i] * torch.matmul(skew_r_oi, J_ti)
        H_w += self.I_0

        H_s = self.total_mass * torch.matmul(skew_r_og, skew_r_og) + H_w
        H_theta = H_w_phi - torch.matmul(skew_r_og, J_tw)

        H_s_inv = torch.linalg.pinv(H_s)

        jacob_bm = torch.zeros((n_samples, n_timesteps, 6, 7)).to(**self.tensor_args)
        jacob_bm[:,:,:3,:] = -(J_tw/self.total_mass + torch.matmul(torch.matmul(skew_r_og, H_s_inv), H_theta))
        jacob_bm[:,:,3:,:] = - torch.matmul(H_s_inv, H_theta)

        # self.logger.info(f"diff: {torch.norm(jacob_bm2 - jacob_bm, p='fro')}")
        
        return jacob_bm


    def make_skew_mat(self, p):
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        row_zero = torch.zeros_like(x)
        return torch.stack([torch.stack([row_zero, -z, y], dim=-1), torch.stack([z, row_zero, -x], dim =-1), torch.stack([-y, x, row_zero], dim=-1)],dim=-2)


    def make_skew_mul_mat(self, p: torch.Tensor) -> torch.Tensor:
        x, y, z = p.unbind(-1)

        r2 = x*x + y*y + z*z

        m00 =  r2 - x*x
        m11 =  r2 - y*y
        m22 =  r2 - z*z
        m01 = - x*y
        m02 = - x*z
        m12 = - y*z

        row0 = torch.stack([m00, m01, m02], dim=-1)
        row1 = torch.stack([m01, m11, m12], dim=-1)
        row2 = torch.stack([m02, m12, m22], dim=-1)
        return torch.stack([row0, row1, row2], dim=-2)
    

    def compute_adj_mat(self, base_pose):
        base_origin = torch.eye((4))

        # Assumption : Base Origin is Origin of World
        del_p = base_pose[:3, 3].clone()
        rot_b = base_pose[:3,:3].clone()
        skew_p = self.make_skew_mat(del_p)
        
        adj_mat = torch.zeros((6,6))
        adj_mat[:3,:3] = adj_mat[3:,3:] = rot_b.clone()
        adj_mat[:3,3:] = torch.matmul(skew_p, rot_b)

        return adj_mat


    def invert_3x3(self, H_s: torch.Tensor) -> torch.Tensor:
        a = H_s[..., 0, 0]; b = H_s[..., 0, 1]; c = H_s[..., 0, 2]
        d = H_s[..., 1, 0]; e = H_s[..., 1, 1]; f = H_s[..., 1, 2]
        g = H_s[..., 2, 0]; h = H_s[..., 2, 1]; i = H_s[..., 2, 2]

        det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)

        C11 =  (e*i - f*h)
        C12 = -(b*i - c*h)
        C13 =  (b*f - c*e)
        C21 = -(d*i - f*g)
        C22 =  (a*i - c*g)
        C23 = -(a*f - c*d)
        C31 =  (d*h - e*g)
        C32 = -(a*h - b*g)
        C33 =  (a*e - b*d)

        adj = torch.stack([
            torch.stack([C11, C21, C31], dim=-1),
            torch.stack([C12, C22, C32], dim=-1),
            torch.stack([C13, C23, C33], dim=-1),
        ], dim=-2)

        H_s_inv = adj / det.unsqueeze(-1).unsqueeze(-1)
        return H_s_inv


    def inverse_cholesky(self, H_s: torch.Tensor):
        with torch.no_grad():
            batch_shape = H_s.shape[:-2]
            H_s_flat = H_s.reshape(-1, 3, 3)
            L, info = torch.linalg.cholesky_ex(H_s_flat)
            
            H_inv = torch.empty_like(H_s_flat)

            spd_mask = (info == 0)

            if spd_mask.any():
                L_spd = L[spd_mask]
                inv_spd = torch.cholesky_inverse(L_spd)
                H_inv[spd_mask] = inv_spd

            nonspd_mask = ~spd_mask
            if nonspd_mask.any():
                H_nonspd = H_s_flat[nonspd_mask]
                inv_nonspd = torch.linalg.pinv(H_nonspd)
                H_inv[nonspd_mask] = inv_nonspd

            H_s_inv = H_inv.reshape(*batch_shape, 3, 3)
        return H_s_inv
    

    def base_update(self, base_pose: torch.Tensor) -> torch.Tensor:
        # Base COM Update 
        # Base = ISS + Link0 로 봄
        # base_pose : tf_mat : torch(4,4)
        base_pose_gpu = base_pose.to(**self.tensor_args)
        link0_com = base_pose_gpu @ self.tf_link0 @ self.com_link0_local
        self.base_com_pose = (link0_com[:3,3].clone() * 243.66 + base_pose_gpu[:3,3] * 100000.0) / (100000.0 + 243.66)


    def compute_system_com(self, com_list: torch.Tensor) -> torch.Tensor:
        n_samples, n_timesteps, _, dof = com_list.shape
        r_com = torch.zeros((n_samples, n_timesteps, 3), **self.tensor_args)
        for i in range(dof):
            r_com += self.mass_list[i] * com_list[...,i].clone()
        r_com += self.base_com_pose.view(1,1,-1) * ((100000.0 + 243.66))
        
        return r_com / self.total_mass