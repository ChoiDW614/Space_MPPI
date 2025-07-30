import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.linalg import cholesky

from rclpy.logging import get_logger

class GaussianSample():
    def __init__(self, n_horizon, n_action, n_buffer, device, seed=0, scale_tril=None, covariance_matrix=None, fixed_samples=False):
        self.logger = get_logger("SampleNoise")
        self.device = device
        self.n_sample = 0
        self.n_horizon = n_horizon
        self.n_action = n_action
        self.n_buffer = n_buffer
        self.n_dims = self.n_horizon * self.n_action
        self.zero_seq = torch.zeros(self.n_horizon * self.n_action, device=self.device)
        self.zero_act_seq = torch.zeros([1, self.n_horizon, self.n_action], device=self.device)
        self.scale_tril = scale_tril
        self.covariance_matrix = covariance_matrix
        self.seed = seed
        self.fixed_samples = fixed_samples
        self.samples = None

        self.action_min = -4.71239
        self.action_max = 4.71239

        self.alpha_mu = 0.1
        self.alpha_sigma = 0.1
        self.mu = torch.zeros([self.n_buffer, self.n_horizon, self.n_action], device=self.device)
        # self.mu = torch.zeros([self.n_horizon, self.n_action], device=self.device)

        if self.scale_tril is None:
            if covariance_matrix is None:
                self.covariance_matrix = torch.eye(self.n_dims, device=self.device)
            self.scale_tril = cholesky(self.covariance_matrix)

        self.mvn = MultivariateNormal(loc=self.zero_seq, scale_tril=self.scale_tril)

        # simple sampling
        self.sigma = torch.eye(self.n_action, device=self.device)
        self.sigma *= 0.1

    
    def get_samples(self, n_sample):
        torch.manual_seed(self.seed)

        samples = self.mvn.sample(sample_shape=[n_sample])
        samples = samples.view(n_sample, self.n_horizon, self.n_action)
        # if filter_smooth:
        #     samples = self.filter_smooth(samples)
        # else:
        #     samples = self.filter_samples(samples)
        self.samples = samples

        return self.samples

    def get_action(self, n_sample, q, seed=None):
        if seed is not None and seed != self.seed:
            self.seed = seed

        if self.n_sample != n_sample or not self.fixed_samples:
            self.n_sample = n_sample

        self.mvn = MultivariateNormal(loc=self.zero_seq, scale_tril=self.scale_tril)
        noise = self.get_samples(n_sample)

        # noise = self.get_samples(n_sample -1)
        # noise = torch.cat((noise, self.zero_act_seq), dim=0)

        # noise = noise.view(self.n_sample, self.n_horizon * self.n_action)
        # noise = torch.matmul(noise, self.scale_tril).view(self.n_sample, self.n_horizon, self.n_action)
        
        act_seq = q.unsqueeze(0) + noise
        act_seq[:,:,0:6] = act_seq[:,:,0:6] * 0.05

        # act_seq = self.scale_act(act_seq, self.action_min, self.action_max)
        return act_seq
    
    def update_distribution(self, weight, action):
        new_mu = torch.einsum('ij, ijkl -> ikl', weight, action)
        self.mu = (1.0 - self.alpha_mu) * self.mu + self.alpha_mu * new_mu

        delta = action - self.mu.unsqueeze(1)

        weighted_delta = torch.sqrt(weight).unsqueeze(-1).unsqueeze(-1) * delta 
        weighted_delta = weighted_delta.view(self.n_buffer, self.n_sample, self.n_dims)
        new_cov = torch.einsum('ijn, ijm -> nm', weighted_delta, weighted_delta)

        new_covariance_matrix = (1.0 - self.alpha_sigma) * self.covariance_matrix + self.alpha_sigma * new_cov
        try:
            self.scale_tril = torch.nan_to_num(cholesky(new_covariance_matrix), nan=1.0)
            self.covariance_matrix = new_covariance_matrix
        except:
            self.reset_covariance()
        return
    
    def reset_covariance(self):
        self.covariance_matrix = torch.eye(self.n_dims, device=self.device)
        self.scale_tril = cholesky(self.covariance_matrix)

    def scale_act(self, act, action_lows, action_highs):
        act = act.clamp(min=action_lows, max=action_highs)
        return act

    def filter_samples(self, samples):
        pass

    def filter_smooth(self, samples):
        pass
    

    def simple_sampling(self, n_sample, seed=None):
        if seed is not None and seed != self.seed:
            self.seed = seed
        if self.n_sample != n_sample or not self.fixed_samples:
            self.n_sample = n_sample
        # torch.manual_seed(self.seed)
        
        # standard_normal_noise = torch.randn(self.n_sample, self.n_horizon, self.n_action, **self.tensor_args)
        # 1. Noise sampling: shape = (n_sample, n_action)
        standard_normal_noise = torch.randn(self.n_samples, self.n_action, **self.tensor_args)
        # 2. Expand over horizon: shape = (n_sample, n_horizon, n_action)
        standard_normal_noise = standard_normal_noise.unsqueeze(1).expand(-1, self.n_horizon, -1)
        self.sigma_matrix = self.sigma.expand(self.n_samples, self.n_horizon, -1, -1)
        noise = torch.matmul(standard_normal_noise.unsqueeze(-2), self.sigma_matrix).squeeze(-2)
        return noise

        return noise