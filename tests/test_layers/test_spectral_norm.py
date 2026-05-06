"""Unit tests for SpectralNormConv2d / SpectralNormLinear (Change 4)."""
import unittest

import torch
import torch.nn.functional as F

from models.layers import SpectralNormConv2d, SpectralNormLinear


class TestSpectralNormConv2d(unittest.TestCase):
    def test_parametrization_registered(self):
        layer = SpectralNormConv2d(3, 8, 3)
        self.assertTrue(hasattr(layer, "parametrizations"))
        self.assertIn("weight", layer.parametrizations)

    def test_strict_mode_is_1_lipschitz(self):
        """With lipschitz_corrected=True, the layer is genuinely 1-Lipschitz
        (Tsuzuku 2018 bound). The default mode is canonical Miyato and is
        NOT 1-Lipschitz for circular convs -- see the next test."""
        from models.layers.lipschitz.spectral_norm import (
            SpectralNormConv2dStrict)
        torch.manual_seed(0)
        layer = SpectralNormConv2dStrict(3, 8, 3, n_power_iters=5)
        layer.train()
        for _ in range(20):
            with torch.no_grad():
                _ = layer(torch.randn(2, 3, 8, 8))
        layer.eval()
        max_ratio = 0.0
        for _ in range(50):
            x = torch.randn(1, 3, 8, 8)
            xp = x + 0.1 * torch.randn_like(x)
            with torch.no_grad():
                y = layer(x)
                yp = layer(xp)
            in_norm = (x - xp).flatten().norm().item()
            out_norm = (y - yp).flatten().norm().item()
            max_ratio = max(max_ratio, out_norm / in_norm)
        self.assertLessEqual(max_ratio, 1.01)

    def test_default_mode_canonical_miyato(self):
        """The default mode (lipschitz_corrected=False) divides by sigma only.
        Verify the output norm is bounded by the matrix bound -- but this
        bound is itself loose for multi-channel circular convs, so the layer
        is NOT 1-Lipschitz; we instead check the (looser) matrix-norm
        property: ||y||_2 <= sqrt(k_h*k_w) * ||x||_2."""
        torch.manual_seed(0)
        layer = SpectralNormConv2d(3, 8, 3, n_power_iters=5)
        layer.train()
        for _ in range(20):
            with torch.no_grad():
                _ = layer(torch.randn(2, 3, 8, 8))
        layer.eval()
        max_ratio = 0.0
        for _ in range(50):
            x = torch.randn(1, 3, 8, 8)
            xp = x + 0.1 * torch.randn_like(x)
            with torch.no_grad():
                y = layer(x)
                yp = layer(xp)
            in_norm = (x - xp).flatten().norm().item()
            out_norm = (y - yp).flatten().norm().item()
            max_ratio = max(max_ratio, out_norm / in_norm)
        # Worst-case bound: sqrt(9) = 3 for a 3x3 conv (kernel area). The
        # actual ratio is data-dependent and typically much smaller; we allow
        # the full theoretical bound here as the loose matrix-norm guarantee.
        self.assertLessEqual(max_ratio, 3.0 + 1e-3)

    def test_u_buffer_evolves_in_train_mode(self):
        # Orthogonal init makes the reshaped weight have all-equal singular
        # values (a degenerate fixed point for power iteration), so we
        # perturb the original weight to give the dominant singular vector
        # a clear direction.
        torch.manual_seed(0)
        layer = SpectralNormConv2d(3, 8, 3, n_power_iters=1)
        layer.train()
        with torch.no_grad():
            layer.parametrizations.weight.original.add_(
                torch.randn_like(layer.parametrizations.weight.original))
        # Reset u to a fresh random vector after the perturbation.
        rescaling = layer.parametrizations.weight[0]
        torch.manual_seed(42)
        u_reset = torch.randn_like(rescaling.u)
        u_reset = u_reset / u_reset.norm()
        rescaling.u.copy_(u_reset)
        u_initial = rescaling.u.clone()
        with torch.no_grad():
            _ = layer(torch.randn(2, 3, 8, 8))
        u_after_one = rescaling.u.clone()
        self.assertFalse(torch.allclose(u_initial, u_after_one, atol=1e-6))

    def test_u_buffer_frozen_in_eval_mode(self):
        torch.manual_seed(0)
        layer = SpectralNormConv2d(3, 8, 3, n_power_iters=1)
        # Warm up.
        layer.train()
        with torch.no_grad():
            _ = layer(torch.randn(2, 3, 8, 8))
        layer.eval()
        u_before = layer.parametrizations.weight[0].u.clone()
        with torch.no_grad():
            _ = layer(torch.randn(2, 3, 8, 8))
            _ = layer(torch.randn(2, 3, 8, 8))
        u_after = layer.parametrizations.weight[0].u.clone()
        self.assertTrue(torch.allclose(u_before, u_after))

    def test_backward_produces_finite_grad(self):
        layer = SpectralNormConv2d(3, 8, 3)
        x = torch.randn(2, 3, 8, 8, requires_grad=True)
        y = layer(x)
        y.sum().backward()
        # Original (un-parametrized) weight is the trainable param.
        original = layer.parametrizations.weight.original
        self.assertIsNotNone(original.grad)
        self.assertTrue(torch.isfinite(original.grad).all())
        self.assertTrue(torch.isfinite(x.grad).all())


class TestSpectralNormLinear(unittest.TestCase):
    def test_forward_lipschitz_tight(self):
        """For Linear, the matrix-norm bound IS the operator norm — the
        Lipschitz constant should equal sigma(W) numerically.
        After warm-up, ||f(x) - f(x')|| / ||x - x'|| should be at most 1
        and reach close to 1 for inputs aligned with the top right singular
        vector."""
        torch.manual_seed(0)
        layer = SpectralNormLinear(16, 8, n_power_iters=20)
        layer.train()
        with torch.no_grad():
            _ = layer(torch.randn(4, 16))
        layer.eval()
        # Direct check: the rescaled weight should have spectral norm <= 1.
        with torch.no_grad():
            W = layer.weight  # parametrized; calling .weight resolves it
            sigma = torch.linalg.svdvals(W)[0].item()
        self.assertLessEqual(sigma, 1.0 + 1e-4)

    def test_backward_produces_finite_grad(self):
        layer = SpectralNormLinear(16, 8)
        x = torch.randn(4, 16, requires_grad=True)
        y = layer(x)
        y.sum().backward()
        original = layer.parametrizations.weight.original
        self.assertIsNotNone(original.grad)
        self.assertTrue(torch.isfinite(original.grad).all())


if __name__ == "__main__":
    unittest.main()
