"""Per-channel 1-Lipschitz N-activation (Prach & Lampert, 2024, eq. 7)."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class NActivation(nn.Module):
    """Three-piece N-shaped activation, slope ±1 on every piece. 1-Lipschitz for any theta.

    `init='absid'` mimics MaxMin's eq. 23 decomposition (sigma = (id, |.|)) over adjacent
    channel pairs and requires even num_channels; the M-rotated layer is then bitwise
    equal to MaxMin at init. `init='identity'` starts as identity; `'random'` uses
    Prach §VI-C.

    Per Prach §V-B6, theta needs a smaller LR than the conv weights. Since the upstream
    optimizer has no per-param-group LRs, we absorb the rescaling here: the parameter
    is `theta_raw` and `theta = lr_scale * theta_raw`. The effective per-step update on
    theta is `lr_scale**2 * eta * dL/dtheta` — the squared factor matches the paper's
    "rescale by 1/10" prescription. Init values are pre-divided by lr_scale.
    """

    def __init__(
        self,
        num_channels: int,
        init: str = 'absid',
        lr_scale: float = 0.1,
    ):
        super().__init__()
        if lr_scale <= 0:
            raise ValueError(f"lr_scale must be positive, got {lr_scale}")
        self.num_channels = int(num_channels)
        self.lr_scale = float(lr_scale)
        self.init = init

        # Pre-divide so `lr_scale * theta_raw` yields the intended starting theta.
        # The actual rescaling — the load-bearing trick — happens in the forward.
        theta1, theta2 = self._initial_thetas(self.num_channels, init)
        self.theta1_raw = nn.Parameter(theta1 / self.lr_scale)
        self.theta2_raw = nn.Parameter(theta2 / self.lr_scale)

    @staticmethod
    def _initial_thetas(num_channels: int, init: str) -> tuple[Tensor, Tensor]:
        if init == 'absid':
            if num_channels % 2 != 0:
                raise ValueError(
                    f"NActivation init='absid' requires even num_channels, "
                    f"got {num_channels}"
                )
            # AbsId: even channel k=2j gets theta1=theta2=0, so N(x)=-x on (-inf, 0]
            # and -x on [0, inf), i.e. plain negation — gauge-equivalent to identity
            # under the rotation M (Prach eq. 23). Odd channel k=2j+1 gets
            # theta1=-100, theta2=0; for any practical x the right knot dominates and
            # N(x) = |x| - 100 (an offset |.|), matching the second sigma of (id, |.|).
            # 100 is just "large enough that |x| << theta1 for typical activations";
            # the offset cancels in the M-rotated comparison.
            theta1 = torch.zeros(num_channels)
            theta2 = torch.zeros(num_channels)
            theta1[1::2] = -100.0
            return theta1, theta2
        if init == 'identity':
            return torch.zeros(num_channels), torch.zeros(num_channels)
        if init == 'random':
            # Prach §VI-C: theta1 ~ U[0, 50], theta2 ~ U[-50, 0].
            u1 = torch.empty(num_channels).uniform_(-5.0, 0.0)
            u2 = torch.empty(num_channels).uniform_(-5.0, 0.0)
            return -10.0 * u1, 10.0 * u2
        raise ValueError(
            f"Unknown NActivation init '{init}'. Expected 'absid', 'identity', or 'random'."
        )

    def _broadcast_thetas(self, ndim: int) -> tuple[Tensor, Tensor]:
        # Effective theta = lr_scale * theta_raw. Since dL/dtheta_raw = lr_scale * dL/dtheta,
        # one SGD step on theta_raw moves theta by lr_scale^2 * eta * dL/dtheta —
        # i.e. lr_scale^2 effective LR, matching Prach's "rescale parameters by 1/10".
        shape = (1, -1) + (1,) * (ndim - 2)
        theta1 = (self.lr_scale * self.theta1_raw).view(*shape)
        theta2 = (self.lr_scale * self.theta2_raw).view(*shape)
        return theta1, theta2

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"NActivation got {x.shape[1]} channels, expected {self.num_channels}"
            )
        theta1, theta2 = self._broadcast_thetas(x.dim())
        # The order of theta1, theta2 in the parameter doesn't matter — the function
        # is defined in terms of the sorted pair (theta_min, theta_max). This makes
        # the parameterisation symmetric and avoids a sign convention.
        theta_min = torch.minimum(theta1, theta2)
        theta_max = torch.maximum(theta1, theta2)
        # Three branches with slope ±1 and matching values at both knots:
        #   x <= theta_min:   x - 2 theta_min   (slope +1, hits -theta_min at x=theta_min)
        #   theta_min<x<theta_max:  -x         (slope -1, hits -theta_min and -theta_max)
        #   x >= theta_max:   x - 2 theta_max   (slope +1, hits -theta_max at x=theta_max)
        # Continuity at both knots gives the "N" shape; |slope|=1 everywhere => 1-Lipschitz.
        return torch.where(
            x <= theta_min,
            x - 2.0 * theta_min,
            torch.where(x >= theta_max, x - 2.0 * theta_max, -x),
        )
