"""Unit tests for NActivation (Change 7, Prach & Lampert 2024 Eq. 7)."""
import math
import unittest

import torch

from models.layers.activations.max_min import MaxMin
from models.layers.activations.n_activation import NActivation


def _rotate_pairs(x: torch.Tensor) -> torch.Tensor:
    """Apply Prach Eq. 24's M = (1/sqrt(2))[[1,1],[1,-1]] to each adjacent
    channel pair of a (N, C, ...) tensor, with C even.

    M is symmetric and self-inverse, so the same call serves as both the
    pre-rotation and the post-rotation in M . sigma . M.
    """
    M = (1.0 / math.sqrt(2.0)) * torch.tensor([[1.0, 1.0], [1.0, -1.0]],
                                              dtype=x.dtype)
    n, c = x.shape[0], x.shape[1]
    spatial = x.shape[2:]
    xr = x.view(n, c // 2, 2, *spatial)
    yr = torch.einsum('ij,ncj...->nci...', M, xr)
    return yr.reshape(n, c, *spatial)


class TestNActivation(unittest.TestCase):
    def test_identity_init_is_identity(self):
        """init='identity' makes the layer the exact identity map."""
        torch.manual_seed(0)
        nact = NActivation(num_channels=6, init='identity')
        x = torch.randn(4, 6, 5, 5) * 4.0
        y = nact(x)
        self.assertEqual((y - x).abs().max().item(), 0.0)

    def test_absid_init_matches_maxmin_under_rotation(self):
        """Prach Eq. 23: M . sigma . M = MaxMin with sigma = (id, |.|).

        With init='absid' the layer realises sigma per channel pair, so
        the M-rotated output is bitwise (modulo float roundoff) equal to
        MaxMin's output. This is the design invariant that justifies the
        adjacent-pair channel grouping; if it ever fails the AbsId init
        order or the math has drifted.
        """
        torch.manual_seed(0)
        nact = NActivation(num_channels=8, init='absid')
        x = torch.randn(4, 8, 5, 5) * 3.0
        y_rot = _rotate_pairs(nact(_rotate_pairs(x)))
        y_mm = MaxMin.forward(x)
        # The rotation introduces O(eps) roundoff via 1/sqrt(2) products.
        self.assertLessEqual((y_rot - y_mm).abs().max().item(), 1e-5)

    def test_lipschitz_one(self):
        """1-Lipschitz for any (theta1, theta2) -- Prach's claim by
        construction (slope +/- 1 on every piece, continuity at knots)."""
        torch.manual_seed(0)
        for init in ('absid', 'identity', 'random'):
            nact = NActivation(num_channels=6, init=init)
            with torch.no_grad():
                # Perturb thetas off init so each channel has a distinct (t1, t2).
                nact.theta1_raw.add_(torch.randn_like(nact.theta1_raw) * 5.0)
                nact.theta2_raw.add_(torch.randn_like(nact.theta2_raw) * 5.0)
            N = 10000
            x = torch.randn(N, 6, 1, 1) * 4.0
            y = torch.randn(N, 6, 1, 1) * 4.0
            fx = nact(x)
            fy = nact(y)
            in_norm = (x - y).flatten(1).norm(p=2, dim=1)
            out_norm = (fx - fy).flatten(1).norm(p=2, dim=1)
            ratio = (out_norm / (in_norm + 1e-12)).max().item()
            self.assertLessEqual(ratio, 1.0 + 1e-5,
                                 msg=f"init={init}: max ratio={ratio}")

    def test_continuous_at_boundaries(self):
        """Boundary values agree: theta_min - 2*theta_min = -theta_min, etc.

        Lipschitz=1 then implies a 2*eps L2 bound on outputs at eps-perturbed
        boundary inputs (mirrors test_householder.py:test_continuous_across_boundary).
        """
        nact = NActivation(num_channels=2, init='identity', lr_scale=1.0)
        # Set theta_min = -0.7, theta_max = 1.3 on channel 0.
        with torch.no_grad():
            nact.theta1_raw[0] = -0.7
            nact.theta2_raw[0] = 1.3
            nact.theta1_raw[1] = 0.0
            nact.theta2_raw[1] = 0.0
        eps = 1e-6
        for boundary in (-0.7, 1.3):
            x_lo = torch.zeros(1, 2, 1, 1)
            x_hi = torch.zeros(1, 2, 1, 1)
            x_lo[0, 0, 0, 0] = boundary - eps
            x_hi[0, 0, 0, 0] = boundary + eps
            y_lo = nact(x_lo)
            y_hi = nact(x_hi)
            diff = (y_hi - y_lo).norm().item()
            self.assertLessEqual(diff, 4 * eps,
                                 msg=f"boundary={boundary}: diff={diff}")

    def test_gradient_flows_to_theta(self):
        torch.manual_seed(0)
        nact = NActivation(num_channels=4, init='absid', lr_scale=0.1)
        # Inputs in a range where both theta1 (~ -100) and theta2 (~0) are
        # both relevant boundaries for at least some samples in the batch
        # we'd need x ~ -100, but here we just need the gradient to flow,
        # which happens through the middle and right pieces.
        x = torch.randn(8, 4, 3, 3, requires_grad=True) * 2.0
        loss = nact(x).pow(2).sum()
        loss.backward()
        self.assertIsNotNone(nact.theta1_raw.grad)
        self.assertIsNotNone(nact.theta2_raw.grad)
        self.assertTrue(torch.isfinite(nact.theta1_raw.grad).all().item())
        self.assertTrue(torch.isfinite(nact.theta2_raw.grad).all().item())
        # At least one of the parameter sets has a non-zero gradient.
        nz = (nact.theta1_raw.grad.abs().sum().item() +
              nact.theta2_raw.grad.abs().sum().item())
        self.assertGreater(nz, 0.0)

    def test_lr_scale_changes_effective_step(self):
        """dL/dtheta_raw = lr_scale * dL/dtheta_effective (chain rule).

        Verified numerically by comparing two layers with different
        lr_scale that compute the same output: gradients on theta_raw must
        scale by the lr_scale ratio.
        """
        torch.manual_seed(0)
        x = torch.randn(8, 4, 3, 3) * 2.0
        # Layer A: lr_scale = 1.0 (no LR rescaling).
        a = NActivation(num_channels=4, init='identity', lr_scale=1.0)
        with torch.no_grad():
            a.theta1_raw.fill_(0.3)
            a.theta2_raw.fill_(-0.5)
        # Layer B: lr_scale = 0.1, raw values 10x larger so effective theta matches.
        b = NActivation(num_channels=4, init='identity', lr_scale=0.1)
        with torch.no_grad():
            b.theta1_raw.fill_(3.0)
            b.theta2_raw.fill_(-5.0)
        # Outputs identical (effective theta = lr_scale * raw is the same).
        ya = a(x)
        yb = b(x)
        self.assertLessEqual((ya - yb).abs().max().item(), 1e-6)
        # But gradients on theta_raw differ by exactly the lr_scale ratio (= 0.1).
        ya.pow(2).sum().backward()
        yb.pow(2).sum().backward()
        ratio = (b.theta1_raw.grad / a.theta1_raw.grad)
        # 0.1 * dL/dtheta_eff / dL/dtheta_eff = 0.1.
        self.assertTrue(torch.allclose(ratio, torch.full_like(ratio, 0.1),
                                       atol=1e-5))

    def test_odd_channels_raises_only_with_absid(self):
        with self.assertRaises(ValueError):
            NActivation(num_channels=5, init='absid')
        # Identity and random init accept odd channel counts.
        NActivation(num_channels=5, init='identity')
        NActivation(num_channels=5, init='random')

    def test_unknown_init_raises(self):
        with self.assertRaises(ValueError):
            NActivation(num_channels=4, init='not_a_real_init')

    def test_wrong_input_channels_raises(self):
        nact = NActivation(num_channels=4, init='identity')
        x = torch.randn(2, 6, 3, 3)
        with self.assertRaises(ValueError):
            nact(x)

    def test_random_init_thetas_in_paper_range(self):
        """Prach Sec. VI-C: theta1 = -10 * U[-5, 0] in [0, 50],
        theta2 = +10 * U[-5, 0] in [-50, 0]."""
        torch.manual_seed(0)
        nact = NActivation(num_channels=128, init='random', lr_scale=1.0)
        t1 = nact.theta1_raw.detach()
        t2 = nact.theta2_raw.detach()
        self.assertTrue((t1 >= 0.0).all().item())
        self.assertTrue((t1 <= 50.0).all().item())
        self.assertTrue((t2 >= -50.0).all().item())
        self.assertTrue((t2 <= 0.0).all().item())


class TestNActivationYAML(unittest.TestCase):
    def test_partial_factory_via_yaml_tag(self):
        import yaml
        from parsers.constructors import DeterministicLoader

        src = """
get_activation: !activation
  name: NActivation
  init: absid
  lr_scale: 0.1
"""
        parsed = yaml.load(src, Loader=DeterministicLoader)
        factory = parsed['get_activation']
        layer = factory(num_channels=8)
        self.assertEqual(layer.num_channels, 8)
        self.assertEqual(layer.theta1_raw.shape, (8,))
        self.assertEqual(layer.theta2_raw.shape, (8,))
        self.assertAlmostEqual(layer.lr_scale, 0.1)
        # AbsId-init layer satisfies the M-rotated MaxMin invariant.
        x = torch.randn(2, 8, 3, 3)
        y_rot = _rotate_pairs(layer(_rotate_pairs(x)))
        y_mm = MaxMin.forward(x)
        self.assertLessEqual((y_rot - y_mm).abs().max().item(), 1e-5)

    def test_scalar_form_resolves_to_class(self):
        import yaml
        from parsers.constructors import DeterministicLoader

        src = "get_activation: !activation NActivation\n"
        parsed = yaml.load(src, Loader=DeterministicLoader)
        layer = parsed['get_activation'](num_channels=8)
        x = torch.randn(1, 8, 1, 1)
        self.assertEqual(layer(x).shape, x.shape)


if __name__ == '__main__':
    unittest.main()
