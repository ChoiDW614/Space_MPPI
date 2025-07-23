# TORCH
import torch

# RCLPY
from rclpy.logging import get_logger

class SavGolFilter:
    def __init__(self):
        self.logger = get_logger("Savgol_Filter")

    def savgol_filter_torch(self, sequence: torch.Tensor, window_size: int, polyorder: int, tensor_args) -> torch.Tensor:
        """
        Apply Savitzky-Golay filter on GPU using PyTorch to each column of the input tensor.

        Args:
            sequence (torch.Tensor): Input sequence of shape (N, 2).
            window_size (int): Window size for the filter (must be odd).
            polyorder (int): Polynomial order (typically 2 or 3).

        Returns:
            torch.Tensor: Smoothed sequence of the same shape as input.
        """
        assert window_size % 2 == 1, "Window size must be odd."
        assert polyorder < window_size, "Polyorder must be less than window size."

        self.tensor_args = tensor_args

        # Split sequence into v and w
        noise1 = sequence[:, 0]
        noise2 = sequence[:, 1]
        noise3 = sequence[:, 2]
        noise4 = sequence[:, 3]
        noise5 = sequence[:, 4]
        noise6 = sequence[:, 5]
        noise7 = sequence[:, 6]
        

        def apply_filter(data: torch.Tensor) -> torch.Tensor:
            """Apply Savitzky-Golay filter to a single column."""
            padding = window_size // 2
            if data.size(0) <= padding:  # Ensure padding is valid
                raise ValueError(f"Padding ({padding}) is too large for data length ({data.size(0)}).")

            # Precompute Savitzky-Golay coefficients
            x = torch.arange(-padding, padding + 1, **self.tensor_args)
            A = torch.stack([x ** i for i in range(polyorder + 1)], dim=1)  # Vandermonde matrix
            ATA_inv = torch.linalg.inv(A.T @ A)
            coeffs = (ATA_inv @ A.T)[0]  # Savitzky-Golay filter coefficients for smoothing

            # Handle padding manually for 1D data
            padded_data = torch.cat([data[:padding].flip(0), data, data[-padding:].flip(0)], dim=0)

            # Apply convolution with the computed coefficients
            smoothed = torch.nn.functional.conv1d(
                padded_data.unsqueeze(0).unsqueeze(0),  # Shape: (1, 1, N + padding*2)
                coeffs.flip(0).view(1, 1, -1),          # Filter coefficients
                padding=0
            ).squeeze(0).squeeze(0)  # Remove batch and channel dimensions
            return smoothed

        # Apply filter to v and w
        smoothed_noise1 = apply_filter(noise1)
        smoothed_noise2 = apply_filter(noise2)
        smoothed_noise3 = apply_filter(noise3)
        smoothed_noise4 = apply_filter(noise4)
        smoothed_noise5 = apply_filter(noise5)
        smoothed_noise6 = apply_filter(noise6)
        smoothed_noise7 = apply_filter(noise7)

        

        # Combine smoothed v and w back into a single tensor
        smoothed_sequence = torch.stack((smoothed_noise1, smoothed_noise2, smoothed_noise3, smoothed_noise4, smoothed_noise5, smoothed_noise6, smoothed_noise7), dim=1)

        return smoothed_sequence
