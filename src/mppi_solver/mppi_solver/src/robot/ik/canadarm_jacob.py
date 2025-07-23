from rclpy.logging import get_logger

import torch

class CanadarmJacob:
    def __init__(self):
        self.logger = get_logger("Canadarm_Jacobian")
        self.axis_list = [2, 0, 2, 2, 2, 0, 2]

        self.mass_list = [105.98, 105.98, 314.98, 279.2, 105.98, 105.98, 243.66]
        self.total_mass = sum(self.mass_list) + 100000.0 + 243.66

        self.inertial_list = torch.zeros((7, 3, 3))
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
        
        tf_link0 = torch.tensor([[1,0,0,0],
                                       [0,-1,0,0],
                                       [0,0,1,3.6],
                                       [0,0,0,1]])
        
        com_link0_local = torch.tensor([[1,0,0,0],
                                        [0,1,0,0],
                                        [0,0,1,0.5],
                                        [0,0,0,1]])
        
        link0_com = torch.matmul(tf_link0,com_link0_local)

        self.base_com_pose = link0_com[:3,3].clone() * 243.66/(100000.0 + 243.66)
        
    def make_skew_mat(self, p):
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        row_zero = torch.zeros_like(x)
        return torch.stack([torch.stack([row_zero, -z, y], dim=-1), torch.stack([z, row_zero, -x], dim =-1), torch.stack([-y, x, row_zero], dim=-1)],dim=-2)


    def compute_jacobian(self, link_pose_list):
        eef = link_pose_list[-2].clone() # -1 은 tip
        n_samples, n_timesteps, _, _ = eef.shape
        device = eef.device
        jacob = torch.zeros((n_samples, n_timesteps, 6, 7), device = device)
        p_eef = eef[:,:,:3,3].clone()

        for i in range(7):
            p = link_pose_list[i][:,:,:3, 3].clone()
            del_p = p_eef - p
            rot = link_pose_list[i][:,:,:3,self.axis_list[i]].clone()
            if i == 4:
                rot *= -1
            jacob[:,:,3:,i] = rot.clone()
            jacob[:,:,:3,i] = torch.cross(rot.clone(), del_p, dim=-1)


        return jacob
    
    def compute_jacobian_cpu(self, link_pose_list):
        eef = link_pose_list[-2].clone()
        jacob = torch.zeros((6,7))
        
        for i in range(7):
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
        # com list[i] : n_samples, n_timestep, 3

        n_samples, n_timesteps, _, _ = link_pose_list[0].shape
        devices = link_pose_list[0].device
        jacob_t = torch.zeros((n_samples, n_timesteps, 3, 7), device=devices)
        jacob_tw = torch.zeros((n_samples, n_timesteps, 3, 7), device = devices)

        jacob_r = torch.zeros((n_samples, n_timesteps, 3, 7), device=devices)
        jacob_rw = torch.zeros((n_samples, n_timesteps, 3, 7), device = devices)
        H_w = torch.zeros((n_samples, n_timesteps, 3, 3), device=devices)
        H_wphi = torch.zeros((n_samples, n_timesteps, 3, 7), device=devices)

        for i in range(7):
            del_p = com_list[i] - link_pose_list[i][:,:,:3,3].clone()
            del_p_skew = self.make_skew_mat(del_p)
            rot = link_pose_list[i][:,:,:3,self.axis_list[i]].clone()
            if i == 4:
                rot *= -1
            jacob_t[:,:,:,i] = torch.cross(rot, del_p, dim=-1)
            jacob_tw += self.mass_list[i] * jacob_t
        
            jacob_r[:,:,:,i] = rot.clone()

            H_w += self.inertial_list[i,:,:].to(device=devices) + self.mass_list[i]*(torch.matmul(del_p_skew.transpose(-1, -2), del_p_skew))
            H_wphi += torch.einsum('ij,ntjl->ntil', self.inertial_list[i,:,:].to(device=devices), jacob_r) + self.mass_list[i] * torch.matmul(del_p_skew, jacob_t)

        system_com = torch.zeros((n_samples, n_timesteps,3), device=devices)
        for i in range(7):
            system_com += com_list[i] * self.mass_list[i]
        system_com/= self.total_mass

        r_og_skew = self.make_skew_mat(system_com - self.base_com_pose.to(device=devices))

        H_s = self.total_mass * (torch.matmul(r_og_skew, r_og_skew)) + H_w
        H_theta = H_wphi - torch.matmul(r_og_skew, jacob_tw)


        jacob_bm = torch.zeros((n_samples, n_timesteps, 6, 7), device=devices)
        
        jacob_bm[:,:,:3,:] = -(jacob_tw/self.total_mass + torch.matmul(torch.matmul(r_og_skew, torch.linalg.pinv(H_s)), H_theta))
        jacob_bm[:,:,3:,:] = -torch.matmul(torch.linalg.pinv(H_s), H_theta)
        
        return jacob_bm