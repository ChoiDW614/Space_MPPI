import os
import math
import yaml
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

import pinocchio as pin

from rclpy.logging import get_logger
from ament_index_python.packages import get_package_share_directory

from mppi_solver.src.robot.urdfFks.urdfFk import URDFForwardKinematics

class MPPI():
    def __init__(self):
        self.logger = get_logger("MPPI")

        # torch env
        os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info('Device: ' + self.device.type)
        torch.manual_seed(time.time_ns())

        self.num_joint = 5

        # Sampling parameters
        self.num_samples = 1024
        self.num_timestep = 256

        self.reset_mu = torch.zeros(self.num_timestep, self.num_joint, device=self.device)
        self.reset_sigma = torch.stack([torch.eye(self.num_joint, device=self.device) for _ in range(self.num_timestep)], dim=0)

        self.mu = self.reset_mu.clone()
        self.sigma = self.reset_sigma.clone()

        # System state

        # self.des_q
        # self.des_qdot

        # self.init_q
        # self.init_base

        package_name = "mppi_solver"
        urdf_file_path = os.path.join(get_package_share_directory(package_name), "models", "curiosity", "curiosity.urdf")

        # with open(urdf_file_path, "r") as file:
        #     urdf = file.read()
        self.fk_curiosity = URDFForwardKinematics(
            urdf_file_path,
            root_link = 'base_link',
            end_links = 'arm_tools',
        )
        self.logger.info("finish")


    def compute_control(self):
        # acc = self.sampling_acceleration()
        fk_result = self.fk_curiosity.fk(torch.tensor([0.02,0.50,-0.32,1.57,0.72], dtype=torch.float64), 'arm_tools')
        # self.logger.info(str(fk_result))
        return
    

    def sampling_acceleration(self):
        try:
            random_generator = torch.distributions.MultivariateNormal(loc=self.mu, covariance_matrix=self.sigma)
        except ValueError:
            self.logger.warn("Covariance_matrix not satisfy the constraint PositiveDefinite")
            self.sigma = self.reset_sigma.clone()
            random_generator = torch.distributions.MultivariateNormal(loc=self.mu, covariance_matrix=self.sigma)
        acc = random_generator.sample((self.num_timestep,))
        return acc
    
