"""Unit tests for Householder activation (Change 6, Singla 2021 Eq. 6)."""
import math
import unittest

import torch

from models.layers.activations.householder import Householder
from models.layers.activations.max_min import MaxMin


class TestHouseholder(unittest.TestCase):
    def test_initial_theta_equals_maxmin(self):
        """At theta=pi/2 default init, output is bitwise equal to MaxMin.

        This is the design invariant that justifies the adjacent-pair channel
        grouping; if it ever fails the pairing or the math has drifted.
        """
        torch.manual_seed(0)
        hh = Householder(num_channels=8)  # default init_theta = pi/2
        x = torch.randn(4, 8, 5, 5)
        y_hh = hh(x)
        y_mm = MaxMin.forward(x)
        # Use a tight float tolerance: the formulas differ by trig identities
        # that introduce O(eps) rounding (sin(pi/4)^2 + cos(pi/4)^2 - 1).
        self.assertLessEqual((y_hh - y_mm).abs().max().item(), 1e-6)

    def test_lipschitz_one(self):
        """Singla's GNP claim: 1-Lipschitz everywhere, for every theta."""
        torch.manual_seed(0)
        for init_theta in (0.0, math.pi / 3, math.pi / 2, 2 * math.pi / 3,
                           math.pi, -math.pi / 4):
            hh = Householder(num_channels=4, init_theta=init_theta)
            with torch.no_grad():
                # Perturb theta off the init so each pair has a different angle.
                hh.theta.add_(torch.randn_like(hh.theta) * 0.5)
            N = 10000
            x = torch.randn(N, 4, 1, 1) * 4.0
            y = torch.randn(N, 4, 1, 1) * 4.0
            fx = hh(x)
            fy = hh(y)
            in_norm = (x - y).flatten(1).norm(p=2, dim=1)
            out_norm = (fx - fy).flatten(1).norm(p=2, dim=1)
            ratio = (out_norm / (in_norm + 1e-12)).max().item()
            self.assertLessEqual(ratio, 1.0 + 1e-5,
                                 msg=f"init_theta={init_theta}: ratio={ratio}")

    def test_norm_preserving_per_pair(self):
        """GNP is stronger than 1-Lipschitz: each pair preserves its norm."""
        torch.manual_seed(0)
        hh = Householder(num_channels=6, init_theta=0.7)
        with torch.no_grad():
            hh.theta.copy_(torch.tensor([0.1, 1.3, -2.5]))
        x = torch.randn(50, 6, 4, 4) * 3.0
        y = hh(x)
        # Per-pair L2 norm: reshape (N, C, H, W) -> (N, C//2, 2, H, W).
        x_pairs = x.view(50, 3, 2, 4, 4)
        y_pairs = y.view(50, 3, 2, 4, 4)
        x_pair_norm = x_pairs.pow(2).sum(dim=2).sqrt()
        y_pair_norm = y_pairs.pow(2).sum(dim=2).sqrt()
        self.assertTrue(torch.allclose(x_pair_norm, y_pair_norm, atol=1e-5))

    def test_continuous_across_boundary(self):
        """At gate=0 the identity and reflected formulas agree (Singla
        continuity argument: H z = z when v^T z = 0)."""
        theta = 0.7
        hh = Householder(num_channels=2, init_theta=theta)
        # Boundary: z1*sin(theta/2) - z2*cos(theta/2) = 0
        #   <=> z1 = z2 * cot(theta/2)
        z2 = 1.3
        z1 = z2 * (math.cos(theta / 2) / math.sin(theta / 2))
        z = torch.tensor([[z1, z2]]).view(1, 2, 1, 1)

        # Approach the boundary from both sides: tiny epsilon perturbations
        # that flip the sign of `gate`. Outputs must agree to first order.
        eps = 1e-6
        z_plus = z.clone()
        z_plus[0, 0, 0, 0] += eps  # increases gate
        z_minus = z.clone()
        z_minus[0, 0, 0, 0] -= eps  # decreases gate
        y_plus = hh(z_plus)
        y_minus = hh(z_minus)
        # Different branches fired, but with Lipschitz=1 the outputs are
        # within 2*eps in L2 norm.
        diff = (y_plus - y_minus).norm().item()
        self.assertLessEqual(diff, 4 * eps)  # 2*eps with margin for fp noise

    def test_jacobian_orthogonal(self):
        """Numerically verify that the per-pair Jacobian is orthogonal in
        both half-spaces (sufficient for GNP)."""
        torch.manual_seed(2)
        for init_theta in (0.0, 0.4, math.pi / 2, 2.1):
            hh = Householder(num_channels=2, init_theta=init_theta)
            for _ in range(5):
                z = torch.randn(2)
                # Avoid the gate=0 hyperplane (zero-measure but where the
                # Jacobian is undefined).
                gate = z[0] * math.sin(init_theta / 2) - z[1] * math.cos(init_theta / 2)
                if gate.abs().item() < 1e-3:
                    continue

                def f(zz):
                    return hh(zz.view(1, 2, 1, 1)).view(2)

                J = torch.autograd.functional.jacobian(f, z)
                eye2 = torch.eye(2)
                self.assertTrue(
                    torch.allclose(J.T @ J, eye2, atol=1e-5),
                    msg=f"theta={init_theta}, z={z.tolist()}, J={J.tolist()}",
                )

    def test_gradient_flows_to_theta(self):
        torch.manual_seed(0)
        hh = Householder(num_channels=4, init_theta=math.pi / 3)
        # Inputs sampled clear of the boundary so dL/dtheta is well-defined.
        x = torch.randn(8, 4, 3, 3, requires_grad=True) * 2.0
        loss = hh(x).pow(2).sum()
        loss.backward()
        self.assertIsNotNone(hh.theta.grad)
        self.assertTrue(torch.isfinite(hh.theta.grad).all().item())
        self.assertGreater(hh.theta.grad.abs().sum().item(), 0.0)

    def test_odd_channels_raises(self):
        with self.assertRaises(ValueError):
            Householder(num_channels=5)

    def test_wrong_input_channels_raises(self):
        hh = Householder(num_channels=4)
        x = torch.randn(2, 6, 3, 3)
        with self.assertRaises(ValueError):
            hh(x)


class TestHouseholderYAML(unittest.TestCase):
    def test_partial_factory_via_yaml_tag(self):
        import yaml
        from parsers.constructors import DeterministicLoader

        src = """
get_activation: !activation
  name: Householder
  init_theta: 1.5707963
"""
        parsed = yaml.load(src, Loader=DeterministicLoader)
        factory = parsed['get_activation']
        layer = factory(num_channels=8)
        self.assertEqual(layer.num_channels, 8)
        self.assertEqual(layer.theta.shape, (4,))
        # init_theta was pi/2, so layer should match MaxMin numerically.
        x = torch.randn(2, 8, 3, 3)
        y_hh = layer(x)
        y_mm = MaxMin.forward(x)
        self.assertLessEqual((y_hh - y_mm).abs().max().item(), 1e-6)

    def test_scalar_form_resolves_to_class(self):
        import yaml
        from parsers.constructors import DeterministicLoader

        src = "get_activation: !activation Householder\n"
        parsed = yaml.load(src, Loader=DeterministicLoader)
        layer = parsed['get_activation'](num_channels=8)
        x = torch.randn(1, 8, 1, 1)
        self.assertEqual(layer(x).shape, x.shape)


if __name__ == '__main__':
    unittest.main()
