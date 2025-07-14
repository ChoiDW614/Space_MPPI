import torch
import torch.nn as nn
from rclpy.logging import get_logger


class MLPRegressionNormDropout(nn.Module):
    def __init__(self):
        super().__init__()
        input_dims: int = 30
        hidden_dim: int = 256
        num_hidden_layers: int = 3
        output_dims: int = 8
        dropout: float = 0.1

        dims = [input_dims] + [hidden_dim] * num_hidden_layers + [output_dims]
        layers: list[nn.Module] = []
        for in_c, out_c in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_c, out_c))
            if out_c != output_dims:
                layers.extend([nn.LayerNorm(out_c),
                               nn.ReLU(),
                               nn.Dropout(dropout)])
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, x.sin(), x.cos()], dim=-1)
        return self.mlp(x)
    

class CollisionAvoidanceCost:
    def __init__(self, params, gamma: float, n_horizon: int, device):
        self.logger = get_logger("Joint_Space_Cost")
        self.device = device
        self.n_horizon = n_horizon
        self.gamma = gamma
        self.model = MLPRegressionNormDropout().to(device=self.device)

    def collision_cost(self,  qSample: torch.Tensor, targets: torch.Tensor):


        return