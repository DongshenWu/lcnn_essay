"""Miyato spectral-normalisation conv/Linear baseline.

Default `lipschitz_corrected=False` is canonical Miyato (divides by sigma(W_mat)
only): NOT 1-Lipschitz for multi-channel circular convs (matrix bound is loose).
`lipschitz_corrected=True` adds the sqrt(k_h * k_w) factor (Tsuzuku et al. 2018,
Lemma 1) for strict 1-Lipschitz, but the resulting per-layer shrinkage prevents
deep classifiers from training. Linear / 1x1 conv: bound is tight either way.
"""
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn.common_types import _size_2_t
from torch.nn.utils.parametrize import register_parametrization

from .spectral_normal_control import power_iteration


_EPS = 1e-12


class SpectralNormRescaling(nn.Module):
    def __init__(
        self,
        weight_shape: Tuple[int, ...],
        n_power_iters: int = 1,
        lipschitz_corrected: bool = False,
    ):
        super().__init__()
        self.n_power_iters = n_power_iters

        if lipschitz_corrected and len(weight_shape) == 4:
            k_h, k_w = weight_shape[2], weight_shape[3]
            kernel_factor = float((k_h * k_w) ** 0.5)
        else:
            kernel_factor = 1.0
        self.register_buffer(
            "kernel_factor", torch.tensor(kernel_factor, dtype=torch.float32))

        # u-buffer warm-started across forwards; updated in train mode only.
        u = torch.randn(weight_shape[0], 1)
        self.register_buffer("u", u / (u.norm() + _EPS))

    def _sigma_only(self, W_mat: Tensor) -> Tensor:
        u = self.u
        v = W_mat.t() @ u
        v = v / (v.norm() + _EPS)
        return (u.t() @ W_mat @ v).reshape(())

    def forward(self, weight: Tensor) -> Tensor:
        W_mat = weight.reshape(weight.shape[0], -1)
        if self.training:
            with torch.no_grad():
                u, _, _ = power_iteration(
                    W_mat.detach(), init_u=self.u, n_iters=self.n_power_iters)
                self.u.copy_(u)
            # u held fixed; sigma keeps grad through W so the divide back-props.
            sigma = self._sigma_only(W_mat)
        else:
            with torch.no_grad():
                sigma = self._sigma_only(W_mat)
        return weight / (sigma * self.kernel_factor + _EPS)


class SpectralNormConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        padding="same",
        padding_mode: str = "circular",
        n_power_iters: int = 1,
        lipschitz_corrected: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size,
            padding=padding, padding_mode=padding_mode, **kwargs,
        )
        nn.init.orthogonal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        register_parametrization(
            self, "weight",
            SpectralNormRescaling(
                tuple(self.weight.shape), n_power_iters, lipschitz_corrected),
        )


class SpectralNormConv2dStrict(SpectralNormConv2d):
    """`SpectralNormConv2d` with `lipschitz_corrected=True` hardwired."""

    def __init__(self, *args, **kwargs):
        kwargs["lipschitz_corrected"] = True
        super().__init__(*args, **kwargs)


class SpectralNormLinear(nn.Linear):
    """For Linear the matrix bound IS the operator norm, so always 1-Lipschitz."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        n_power_iters: int = 1,
        lipschitz_corrected: bool = False,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        nn.init.orthogonal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        register_parametrization(
            self, "weight",
            SpectralNormRescaling(
                tuple(self.weight.shape), n_power_iters, lipschitz_corrected),
        )
