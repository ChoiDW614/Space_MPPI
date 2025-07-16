from rclpy.logging import get_logger

import torch

class CanadarmJacob:
    def __init__(self):
        self.logger = get_logger("Canadarm_Jacobian")
        self.axis_list = [2, 0, 2, 2, 2, 0, 2]


    # LOCAL WORLD ALIGNED
    # Adj Mat (based on base pose) multiply


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
    
    def compute_base_jacob(self, base_pose):
        # base jacobian linear : j_base_linear + skew(p_eef - p) * j_base_angular
        # base jacobian angular : j_base_angular
        pass