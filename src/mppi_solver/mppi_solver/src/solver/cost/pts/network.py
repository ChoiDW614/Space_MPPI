import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)
    

class MLPWithResidualNorm(nn.Module):
    def __init__(self):
        super().__init__()
        input_dims = 30
        hidden_dim: int = 256
        num_blocks: int = 2
        output_dims: int = 8

        self.proj    = nn.Linear(input_dims, hidden_dim)
        self.norm0   = nn.LayerNorm(hidden_dim)
        self.act0    = nn.ReLU(inplace=True)
        self.blocks  = nn.ModuleList([
            ResidualBlock(hidden_dim)
            for _ in range(num_blocks)
        ])
        self.out_fc  = nn.Linear(hidden_dim, output_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_enc = torch.cat([x, torch.sin(x), torch.cos(x)], dim=-1)

        h = self.proj(x_enc)
        h = self.norm0(h)
        h = self.act0(h)

        for block in self.blocks:
            h = block(h)

        return self.out_fc(h)


class MLPRegressionNormDropout(nn.Module):
    def __init__(self):
        super().__init__()
        input_dims: int = 30
        hidden_dim: int = 256
        num_hidden_layers: int = 4
        output_dims: int = 8
        dropout: float = 0.1

        mlp_arr: list[list[int]] = []
        mlp_arr.append(([hidden_dim] * num_hidden_layers).copy())
        mlp_arr[-1].append(output_dims)
        mlp_arr[0].insert(0, input_dims)

        self.layers = nn.ModuleList()
        channels = mlp_arr[0]
        blocks: list[nn.Module] = []
        for i in range(len(channels) - 1):
            in_c, out_c = channels[i], channels[i+1]
            blocks.append(nn.Linear(in_c, out_c, bias=True))
            if i < len(channels) - 2:
                blocks.append(nn.ReLU())
                blocks.append(nn.Dropout(dropout))
        self.layers.append(nn.Sequential(*blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_enc = torch.cat([x, torch.sin(x), torch.cos(x)], dim=-1)
        return self.layers[0](x_enc)
    

# -----------------------------------------------------------------


class ResidualELUBlock(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ELU(inplace=True),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ELU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)
    

class MLPWithResidualNormELU(nn.Module):
    def __init__(self):
        super().__init__()
        input_dims = 30
        hidden_dim: int = 256
        num_blocks: int = 2
        output_dims: int = 8

        self.proj    = nn.Linear(input_dims, hidden_dim)
        self.norm0   = nn.LayerNorm(hidden_dim)
        self.act0    = nn.ELU(inplace=True)
        self.blocks  = nn.ModuleList([
            ResidualELUBlock(hidden_dim)
            for _ in range(num_blocks)
        ])
        self.out_fc  = nn.Linear(hidden_dim, output_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_enc = torch.cat([x, torch.sin(x), torch.cos(x)], dim=-1)

        h = self.proj(x_enc)
        h = self.norm0(h)
        h = self.act0(h)

        for block in self.blocks:
            h = block(h)

        return self.out_fc(h)


class MLPRegressionNormDropoutELU(nn.Module):
    def __init__(self):
        super().__init__()
        input_dims: int = 30
        hidden_dim: int = 256
        num_hidden_layers: int = 4
        output_dims: int = 8
        dropout: float = 0.1

        mlp_arr: list[list[int]] = []
        mlp_arr.append(([hidden_dim] * num_hidden_layers).copy())
        mlp_arr[-1].append(output_dims)
        mlp_arr[0].insert(0, input_dims)

        self.layers = nn.ModuleList()
        channels = mlp_arr[0]
        blocks: list[nn.Module] = []
        for i in range(len(channels) - 1):
            in_c, out_c = channels[i], channels[i+1]
            blocks.append(nn.Linear(in_c, out_c, bias=True))
            if i < len(channels) - 2:
                blocks.append(nn.ELU())
                blocks.append(nn.Dropout(dropout))
        self.layers.append(nn.Sequential(*blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_enc = torch.cat([x, torch.sin(x), torch.cos(x)], dim=-1)
        return self.layers[0](x_enc)
    