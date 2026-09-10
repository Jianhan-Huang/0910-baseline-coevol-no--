import torch
import torch.nn as nn


class OperatorNet(nn.Module):
    """Branch network wrapper for PDE tasks.

    Wraps a branch model with an output projection head that maps per-point
    features to one or more predicted physical-state channels.
    """

    def __init__(self, branch, num_basis=128, resolution=64, out_channels=1):
        super().__init__()
        self.branch = branch
        self.resolution = resolution
        self.out_channels = out_channels

        self.output_proj = nn.Sequential(
            nn.Linear(num_basis, num_basis * 4),
            nn.GELU(),
            nn.Linear(num_basis * 4, out_channels),
        )

    def forward(self, u_, x_, t_=None):
        weight = self.branch(u_, x_, t_)
        out = self.output_proj(weight)
        if self.out_channels == 1:
            return out.squeeze(-1).reshape(out.shape[0], -1)
        return out.reshape(out.shape[0], -1, self.out_channels)
