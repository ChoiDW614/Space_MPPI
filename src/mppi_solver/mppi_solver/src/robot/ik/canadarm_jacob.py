from rclpy.logging import get_logger
from time import time
import torch

class CanadarmJacob:
    def __init__(self, params, tensor_args):
        self.logger = get_logger("Canadarm_Jacobian")
        self.tensor_args = tensor_args
        self.n_samples = params['mppi']['sample']
        self.n_horizon = params['mppi']['horizon']
        self.n_action = params['mppi']['action']

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
        
        tf_link0 = torch.tensor([[1, 0,   0, 0],
                                 [0,-1,   0, 0],
                                 [0, 0, 1.3, 6],
                                 [0, 0,   0, 1]], **self.tensor_args)
        
        com_link0_local = torch.tensor([[1, 0, 0,   0],
                                        [0, 1, 0,   0],
                                        [0, 0, 1, 0.5],
                                        [0, 0, 0,   1]], **self.tensor_args)
        
        link0_com = torch.matmul(tf_link0, com_link0_local)

        self.base_com_pose = link0_com[:3,3].clone() * 243.66 / (100000.0 + 243.66)

        # for optimization
        self.sum_mass = torch.cumsum(self.mass_list.flip(0), dim=0).flip(0).view(1, 1, 1, self.n_action)
        self.sum_inertial = torch.cumsum(self.inertial_list.flip(0), dim=0).flip(0)
        self.H_wphi_mask = torch.tril(torch.ones(7,7, dtype=torch.bool, device=self.tensor_args['device'])).view(1,1,self.n_action,1,self.n_action)
        

    def make_skew_mat(self, p):
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        row_zero = torch.zeros_like(x)
        return torch.stack([torch.stack([row_zero, -z, y], dim=-1), torch.stack([z, row_zero, -x], dim =-1), torch.stack([-y, x, row_zero], dim=-1)],dim=-2)


    def compute_jacobian(self, link_pose_list):
        link_pose_list = torch.stack(link_pose_list, dim=-1).to(**self.tensor_args)
        n_samples, n_horizon, _, _, _ = link_pose_list.shape
        jacob = torch.zeros((n_samples, n_horizon, 6, self.n_action), **self.tensor_args)

        del_p = link_pose_list[:,:,:3,3,-2].unsqueeze(-1) - link_pose_list[:,:,:3,3,:self.n_action]
        rot = link_pose_list[:,:,:3, self.axis_list, self.link_list]
        rot[...,4] *= -1

        jacob[:,:,:3,:] = torch.cross(rot, del_p, dim=2)
        jacob[:,:,3:,:] = rot.clone()
        return jacob
    

    def compute_jacobian_cpu(self, link_pose_list):
        eef = link_pose_list[-2].clone()
        jacob = torch.zeros((6, self.n_action))
        
        for i in range(self.n_action):
            p = link_pose_list[i][:3, 3].clone()

            del_p = eef[:3, 3].clone() - p
            rot = link_pose_list[i][:3, self.axis_list[i]].clone()
            if i == 4:
                rot *= -1
            # self.logger.info(f"{i}-th : {del_p}")

            jacob[3:,i] = rot.clone()
            jacob[:3,i] = torch.cross(rot.clone(), del_p, dim=-1)
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

    # 이게 일단은 안쓰이나봄?
    # compute jacobian considering floating base
    def compute_jacob_floating(self, base_pose, jacobian):
        adj_mat = self.compute_adj_mat(base_pose)
        adj_mat = adj_mat.to(jacobian.device)
        manip_jacob = jacobian.clone()

        return torch.matmul(adj_mat, manip_jacob)
    

    def compute_jacob_bm(self, com_list, link_pose_list):
        start_time = time()

        with torch.no_grad():
            com_list = torch.stack(com_list, dim=-1).to(**self.tensor_args)
            link_pose_list = torch.stack(link_pose_list, dim=-1).to(**self.tensor_args)
            n_samples, n_horizon, _, _ = com_list.shape

            del_p = com_list[:,:,:,:self.n_action] - link_pose_list[:,:,:3,3,:self.n_action]
            del_p_skew = self.make_skew_mat(del_p.permute(0,1,3,2))
            rot = link_pose_list[:,:,:3,self.axis_list,self.link_list]
            rot[...,4] *= -1

            jacob_t = torch.cross(rot, del_p, dim=2)
            jacob_r = rot.clone()

            jacob_tw = jacob_t * self.sum_mass

            H_w = torch.sum(self.inertial_list.view(1,1,self.n_action,3,3) \
                            + self.mass_list.view(1,1,self.n_action,1,1) * (del_p_skew.transpose(-1,-2) @ del_p_skew), dim=2)
            H_wphi = torch.einsum('aij,shja->shia', self.sum_inertial, jacob_r) + \
                    torch.sum(torch.einsum('shaij,shjk->shaik', del_p_skew, jacob_t) * self.mass_list.view(1,1,self.n_action,1,1) * self.H_wphi_mask, dim=2)

            system_com = torch.einsum('ntci,i->ntc', com_list, self.mass_list) / self.total_mass
            r_og_skew = self.make_skew_mat(system_com - self.base_com_pose)

            H_s = self.total_mass * (torch.matmul(r_og_skew, r_og_skew)) + H_w
            H_theta = H_wphi - torch.matmul(r_og_skew, jacob_tw)

            jacob_bm = torch.zeros((n_samples, n_horizon, 6, 7), **self.tensor_args)

            H_s_inv = self.invert_3x3(H_s)

            jacob_bm[:,:,:3,:] = -(jacob_tw / self.total_mass + torch.matmul(torch.matmul(r_og_skew, H_s_inv), H_theta))
            jacob_bm[:,:,3:,:] = -torch.matmul(H_s_inv, H_theta)

        return jacob_bm

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

    