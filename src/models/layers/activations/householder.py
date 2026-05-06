"""Per-channel-pair learnable 1-Lipschitz Householder activation (Singla et al., 2021, eq. 6)."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class Householder(nn.Module):
    """Order-1 Householder activation: identity-or-reflect over each adjacent channel pair.

    Both Jacobians are orthogonal so the activation is 1-Lipschitz for any theta.
    Default init_theta = pi/2 makes it bitwise equivalent to MaxMin (verified in tests).
    Adjacent-pair convention (matching MaxMin) is gauge-equivalent to Singla's split-half.
    """

    def __init__(self, num_channels: int, init_theta: float = math.pi / 2):
        super().__init__()
        if num_channels % 2 != 0:
            raise ValueError(
                f"Householder requires an even num_channels, got {num_channels}"
            )
        self.num_channels = int(num_channels)
        self.theta = nn.Parameter(
            torch.full((num_channels // 2,), float(init_theta))
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"Householder got {x.shape[1]} channels, expected {self.num_channels}"
            )

        # Group the C channels into C/2 adjacent pairs by inserting a size-2 axis.
        # z1 and z2 are the two channels of every pair, broadcast across spatial dims.
        in_size = x.size()
        x_rs = x.view(in_size[0], in_size[1] // 2, 2, *in_size[2:])
        z1, z2 = x_rs.select(2, 0), x_rs.select(2, 1)

        # theta is one parameter per pair; reshape to (1, C/2, 1, 1, ...) so it
        # broadcasts against (N, C/2, H, W) without copying.
        theta = self.theta.view((1, -1) + (1,) * (x.dim() - 2))
        sin_half, cos_half = torch.sin(theta / 2), torch.cos(theta / 2)
        sin_t, cos_t = torch.sin(theta), torch.cos(theta)

        # Singla eq. 6: pick identity when the input lies on the half-space defined by
        # the half-angle hyperplane normal v = (sin(theta/2), -cos(theta/2)); else
        # reflect across that hyperplane. Both branches preserve the per-pair L2 norm
        # — the Jacobian on each pair is either I or a Householder reflection R,
        # both orthogonal, so the layer is 1-Lipschitz for any real theta.
        identity_branch = (z1 * sin_half - z2 * cos_half) > 0
        z1_reflected = cos_t * z1 + sin_t * z2
        z2_reflected = sin_t * z1 - cos_t * z2

        z1_out = torch.where(identity_branch, z1, z1_reflected)
        z2_out = torch.where(identity_branch, z2, z2_reflected)

        # Re-interleave the pair axis back into the channel axis.
        return torch.stack([z1_out, z2_out], dim=2).view(*in_size)
