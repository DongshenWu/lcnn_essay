"""Per-channel 1-Lipschitz learnable linear-spline activation (Ducotterd et al., 2024)."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.parametrize import register_parametrization


class SplineProj(nn.Module):
    """Project knot coefficients onto {c : |c_{k+1} - c_k| <= dx} (Ducotterd eq. 13).

    Matrix-free form: clip first differences, reconstruct via cumsum, restore mean.
    """

    def __init__(self, dx: float):
        super().__init__()
        self.dx = float(dx)

    def forward(self, c: Tensor) -> Tensor:
        # 1-Lipschitz across the spline <=> every consecutive slope (c_{k+1}-c_k)/dx
        # in [-1, 1]. Clip the first differences in the gradient domain, then integrate
        # back via cumsum. Cumsum loses the offset, so we re-center on the original mean
        # — keeps the projection mean-preserving and idempotent.
        mean = c.mean(dim=-1, keepdim=True)
        diffs = (c[..., 1:] - c[..., :-1]).clamp(-self.dx, self.dx)
        zeros = torch.zeros_like(diffs[..., :1])
        recon = torch.cat([zeros, diffs.cumsum(dim=-1)], dim=-1)
        return recon - recon.mean(dim=-1, keepdim=True) + mean


class LinearSpline(nn.Module):
    """Per-channel piecewise-linear activation with K knots in [-T, T], 1-Lipschitz via SplineProj.

    `init='maxmin'` is an alias for 'absolute_value': MaxMin proper is a multi-channel
    sort and has no component-wise analogue. Outside [-T, T] the spline extrapolates
    linearly using the slope of the last interval. `tv_weight` exposes a second-order
    TV penalty via `tv2_penalty()` — callers must add it to the loss themselves.
    """

    def __init__(
        self,
        num_channels: int,
        num_knots: int = 21,
        range: float = 3.0,
        init: str = 'identity',
        tv_weight: float = 0.0,
    ):
        super().__init__()
        if num_knots < 3:
            raise ValueError(f"num_knots must be >= 3, got {num_knots}")
        T = float(range)
        if T <= 0:
            raise ValueError(f"range must be positive, got {T}")
        self.num_channels = int(num_channels)
        self.num_knots = int(num_knots)
        self.T = T
        self.dx = 2.0 * T / (num_knots - 1)
        self.tv_weight = float(tv_weight)
        self.init = init

        self.register_buffer('knot_xs', torch.linspace(-T, T, num_knots))

        coef = torch.empty(self.num_channels, self.num_knots)
        self._initialize_coef(coef, init)
        self.coef = nn.Parameter(coef)

        register_parametrization(self, 'coef', SplineProj(self.dx))

    def _initialize_coef(self, coef: Tensor, init: str) -> None:
        xs = torch.linspace(-self.T, self.T, self.num_knots)
        if init == 'identity':
            init_vals = xs
        elif init in ('absolute_value', 'maxmin', 'abs'):
            init_vals = xs.abs()
        elif init == 'relu':
            init_vals = xs.clamp(min=0.0)
        else:
            raise ValueError(
                f"Unknown LinearSpline init '{init}'. "
                f"Expected one of: identity, absolute_value, maxmin, abs, relu."
            )
        coef.copy_(init_vals.unsqueeze(0).expand(self.num_channels, -1).clone())

    def tv2_penalty(self) -> Tensor:
        # Discrete second derivative (c_{k+1} - 2 c_k + c_{k-1}) approximates dx^2 * f''.
        # Summing |.| / dx gives a TV-like penalty on f'', driving the spline toward
        # piecewise-affine functions with few linear regions (Ducotterd §3.3).
        c = self.coef
        d2 = c[:, 2:] - 2.0 * c[:, 1:-1] + c[:, :-2]
        return self.tv_weight * d2.abs().sum() / self.dx

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"LinearSpline got {x.shape[1]} channels, expected {self.num_channels}"
            )

        # Locate each x in the knot grid: bin index k and fractional offset t in [0, 1).
        # Clamping k to [0, K-2] picks the *last* interval for any x outside [-T, T]
        # — the slope of that interval extends linearly, giving the linear extrapolation
        # that keeps the function 1-Lipschitz globally.
        u = (x + self.T) / self.dx
        k = u.floor().clamp(0, self.num_knots - 2).long()
        t = u - k.to(u.dtype)

        # Flatten the spatial dims; we'll gather per-channel coefficients in 1D, then
        # reshape back. Each channel has its own knot vector, so the gather happens
        # along the knot axis after broadcasting coef to (N, C, K).
        N, C = x.shape[0], self.num_channels
        k_flat = k.reshape(N, C, -1)
        t_flat = t.reshape(N, C, -1)

        coef_expanded = self.coef.unsqueeze(0).expand(N, C, self.num_knots)
        c_k = coef_expanded.gather(2, k_flat)
        c_kp1 = coef_expanded.gather(2, k_flat + 1)
        # Linear interpolation between the two enclosing knots.
        return (c_k + t_flat * (c_kp1 - c_k)).view_as(x)
