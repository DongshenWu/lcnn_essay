"""Unit tests for LinearSpline / SplineProj (Change 3, LLS activation)."""
import unittest

import torch

from models.layers.activations.learnable_linear_spline import (
    LinearSpline, SplineProj)


class TestSplineProj(unittest.TestCase):
    def test_idempotent_on_feasible_input(self):
        """SplineProj is a projection: applying twice == applying once."""
        torch.manual_seed(0)
        dx = 0.4
        proj = SplineProj(dx)
        # A feasible vector: identity-like coefficients with |dc| <= dx.
        c = torch.linspace(-2.0, 2.0, 11).unsqueeze(0).expand(3, -1).clone()
        c1 = proj(c)
        c2 = proj(c1)
        self.assertTrue(torch.allclose(c1, c2, atol=1e-6))

    def test_clips_infeasible_differences(self):
        torch.manual_seed(0)
        dx = 0.5
        proj = SplineProj(dx)
        # An infeasible vector with large jumps.
        c = torch.tensor([[0.0, 5.0, -3.0, 8.0, 1.0]])
        c_proj = proj(c)
        diffs = c_proj[:, 1:] - c_proj[:, :-1]
        self.assertLessEqual(diffs.abs().max().item(), dx + 1e-6)

    def test_preserves_mean(self):
        torch.manual_seed(0)
        proj = SplineProj(dx=0.3)
        c = torch.randn(5, 9) * 3.0
        c_proj = proj(c)
        self.assertTrue(torch.allclose(
            c.mean(dim=-1), c_proj.mean(dim=-1), atol=1e-6))


class TestLinearSpline(unittest.TestCase):
    def test_parametrization_registered(self):
        lls = LinearSpline(num_channels=4, num_knots=7, range=2.0)
        self.assertTrue(hasattr(lls, 'parametrizations'))
        self.assertIn('coef', lls.parametrizations)

    def test_coefficient_constraint_enforced(self):
        """After SplineProj, |c_{k+1} - c_k| <= dx for any internal state."""
        torch.manual_seed(0)
        lls = LinearSpline(num_channels=3, num_knots=11, range=3.0)
        # Inject infeasible values into the unconstrained backing parameter.
        with torch.no_grad():
            lls.parametrizations.coef.original.copy_(
                torch.randn_like(lls.parametrizations.coef.original) * 5.0)
        c = lls.coef
        diffs = c[:, 1:] - c[:, :-1]
        self.assertLessEqual(diffs.abs().max().item(), lls.dx + 1e-5)

    def test_identity_init_is_identity(self):
        """SPEC criterion 3: with init='identity' and no training, LLS is
        numerically equal to the identity on inputs in [-T, T]."""
        torch.manual_seed(0)
        T = 3.0
        lls = LinearSpline(num_channels=4, num_knots=21, range=T,
                           init='identity')
        x = torch.linspace(-T + 1e-3, T - 1e-3, 200).view(1, 1, 1, -1).expand(
            2, 4, 1, 200).contiguous()
        y = lls(x)
        self.assertLessEqual((y - x).abs().max().item(), 1e-5)

    def test_lipschitz_one(self):
        """SPEC criterion 1: random pairs satisfy |f(x) - f(y)| <= |x - y|.

        Tighter than Monte Carlo on inputs alone -- we also stress-test by
        starting from a non-degenerate, perturbed coefficient state.
        """
        torch.manual_seed(0)
        lls = LinearSpline(num_channels=4, num_knots=11, range=3.0,
                           init='absolute_value')
        with torch.no_grad():
            lls.parametrizations.coef.original.add_(
                torch.randn_like(lls.parametrizations.coef.original) * 2.0)

        N = 10000
        x = torch.randn(N, 4, 1, 1) * 4.0
        y = torch.randn(N, 4, 1, 1) * 4.0
        fx = lls(x)
        fy = lls(y)
        in_norm = (x - y).flatten(1).norm(p=2, dim=1)
        out_norm = (fx - fy).flatten(1).norm(p=2, dim=1)
        ratio = (out_norm / (in_norm + 1e-12)).max().item()
        self.assertLessEqual(ratio, 1.0 + 1e-5)

    def test_gradcheck(self):
        """SPEC criterion 2: gradcheck w.r.t. x passes on a tiny instance."""
        torch.manual_seed(1)
        lls = LinearSpline(num_channels=4, num_knots=7, range=2.0,
                           init='identity')
        with torch.no_grad():
            lls.parametrizations.coef.original.add_(
                torch.randn_like(lls.parametrizations.coef.original) * 0.05)
        lls = lls.double()
        # Pick x clear of knot boundaries to avoid kinks in the
        # piecewise-linear forward.
        x = torch.tensor([[0.13, -0.71, 0.42, 1.27]],
                         dtype=torch.double, requires_grad=True)
        ok = torch.autograd.gradcheck(
            lambda z: lls(z.unsqueeze(-1).unsqueeze(-1)).sum(),
            (x,), eps=1e-4, atol=1e-3)
        self.assertTrue(ok)

    def test_gradient_flows_to_coefficients(self):
        torch.manual_seed(0)
        lls = LinearSpline(num_channels=4, num_knots=11, range=3.0,
                           init='absolute_value')
        x = torch.randn(2, 4, 8, 8, requires_grad=True)
        y = lls(x).sum()
        y.backward()
        coef_grad = lls.parametrizations.coef.original.grad
        self.assertIsNotNone(coef_grad)
        self.assertTrue(torch.isfinite(coef_grad).all().item())
        self.assertGreater(coef_grad.abs().sum().item(), 0.0)

    def test_extrapolates_linearly_outside(self):
        """Outside [-T, T] the spline extrapolates with the slope of the
        last interval; identity init has slope 1 everywhere."""
        T = 2.0
        lls = LinearSpline(num_channels=2, num_knots=11, range=T,
                           init='identity')
        x = torch.tensor([[-5.0, -T, 0.0, T, 5.0]]).unsqueeze(-1).unsqueeze(
            -1).expand(1, 2, 5, 1, 1).reshape(1, 2, 5, 1)
        y = lls(x)
        # With identity init the spline is the identity everywhere.
        self.assertLessEqual((y - x).abs().max().item(), 1e-5)

    def test_tv2_penalty_is_scalar_and_non_negative(self):
        torch.manual_seed(0)
        lls = LinearSpline(num_channels=3, num_knots=11, range=3.0,
                           init='absolute_value', tv_weight=0.5)
        pen = lls.tv2_penalty()
        self.assertEqual(pen.dim(), 0)
        self.assertGreaterEqual(pen.item(), 0.0)


class TestLinearSplineYAML(unittest.TestCase):
    def test_partial_factory_via_yaml_tag(self):
        import yaml
        from parsers.constructors import DeterministicLoader

        src = """
get_activation: !activation
  name: LinearSpline
  num_knots: 7
  range: 2.0
  init: identity
"""
        parsed = yaml.load(src, Loader=DeterministicLoader)
        factory = parsed['get_activation']
        layer = factory(num_channels=8)
        self.assertEqual(layer.num_channels, 8)
        self.assertEqual(layer.num_knots, 7)
        self.assertAlmostEqual(layer.T, 2.0)

    def test_scalar_form_resolves_to_class(self):
        import yaml
        from parsers.constructors import DeterministicLoader

        src = "get_activation: !activation MaxMin\n"
        parsed = yaml.load(src, Loader=DeterministicLoader)
        # Scalar form gives back the class itself; instantiate as the model would.
        layer = parsed['get_activation'](num_channels=8)
        x = torch.randn(1, 8, 1, 1)
        self.assertEqual(layer(x).shape, x.shape)


if __name__ == '__main__':
    unittest.main()
