# TORCH
import torch

# RCLPY
from rclpy.logging import get_logger

class MovingAverageFilter:
    def __init__(self):
        self.logger = get_logger("Moving_Avg_Filter")

    def moving_average_filter(self, xx: torch.Tensor, window_size: int) -> torch.Tensor:
        """
        Apply moving average filter for smoothing input sequence.

        Args:
            xx (torch.Tensor): Input tensor of shape (N, dim), where N is the sequence length.
            window_size (int): Size of the moving average window.

        Returns:
            torch.Tensor: Smoothed tensor of the same shape as xx.
        """
        N, dim = xx.shape  # N: sequence length, dim: number of dimensions

        # Reshape xx to (batch_size=1, channels=dim, sequence_length=N)
        xx = xx.t().unsqueeze(0)  # Shape: (1, dim, N)

        # Create the filter weights for convolution
        b = torch.ones((dim, 1, window_size), device=xx.device) / window_size  # Shape: (dim, 1, window_size)

        # Adjust padding to ensure output length matches input length
        padding_left = window_size // 2
        padding_right = window_size - padding_left - 1

        xx_padded = torch.nn.functional.pad(xx, (padding_left, padding_right), mode='reflect')  # Shape: (1, dim, N + padding_left + padding_right)

        # Perform convolution using groups to apply the filter independently to each dimension
        xx_mean = torch.nn.functional.conv1d(xx_padded, b, groups=dim)  # Shape: (1, dim, N)

        # Reshape back to (N, dim)
        xx_mean = xx_mean.squeeze(0).t()  # Shape: (N, dim)

        # Edge correction to compensate for the convolution effect at the boundaries
        n_conv = (window_size + 1) // 2  # Equivalent to math.ceil(window_size / 2)

        # Correct the first element
        factor0 = window_size / n_conv
        xx_mean[0, :] *= factor0

        if n_conv > 1:
            # Indices for the rest of the elements to correct
            i_range = torch.arange(1, n_conv, device=xx.device)  # [1, 2, ..., n_conv - 1]

            # Factors for the beginning of the sequence
            factor_start = window_size / (i_range + n_conv)  # Shape: (n_conv - 1,)
            xx_mean[1:n_conv, :] *= factor_start.unsqueeze(1)  # Apply factors to xx_mean[1:n_conv, :]

            # Factors for the end of the sequence
            denom_end = i_range + n_conv - (window_size % 2)
            factor_end = window_size / denom_end  # Shape: (n_conv - 1,)
            xx_mean[-n_conv+1:, :] *= factor_end.flip(0).unsqueeze(1)  # Apply factors to xx_mean[-(n_conv-1):, :]

        return xx_mean