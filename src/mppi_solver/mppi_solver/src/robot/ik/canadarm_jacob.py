from rclpy.logging import get_logger

import torch

class CanadarmJacob:
    def __init__(self):
        self.logger = get_logger("Canadarm_Jacobian")
        self.axis_list = [2, 0, 2, 2, 2, 0, 2]


    def compute_jacobian(self, link_pose_list):
        eef = link_pose_list[-1].clone()
        n_samples, n_timesteps, _, _ = eef.shape
        device = eef.device
        jacob = torch.zeros((n_samples, n_timesteps, 6, 7), device = device)
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