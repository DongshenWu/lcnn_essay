import unittest

import torch

from trainer.metrics import MulticlassHingeWithMargin


class TestMulticlassHingeWithMargin(unittest.TestCase):
    def test_zero_when_margin_satisfied(self):
        # f_y exceeds every other logit by at least margin -> loss is 0.
        logits = torch.tensor([[5., 1., 2.],
                               [1., 5., 3.]])
        labels = torch.tensor([0, 1])
        loss = MulticlassHingeWithMargin(margin=1.0)(logits, labels)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_hand_computed_value(self):
        # Sample 0: max_other=10, f_y=5, kappa=1 -> max(0, 1+10-5) = 6
        # Sample 1: max_other=3,  f_y=5, kappa=1 -> max(0, 1+3-5)  = 0
        # mean = 3.0
        logits = torch.tensor([[5., 10., 2.],
                               [1.,  5., 3.]])
        labels = torch.tensor([0, 1])
        loss = MulticlassHingeWithMargin(margin=1.0)(logits, labels)
        self.assertAlmostEqual(loss.item(), 3.0, places=6)

    def test_non_negative_on_random_inputs(self):
        torch.manual_seed(0)
        logits = torch.randn(64, 10)
        labels = torch.randint(0, 10, (64,))
        loss = MulticlassHingeWithMargin(margin=0.4)(logits, labels)
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_backward_produces_finite_grad(self):
        logits = torch.randn(8, 5, requires_grad=True)
        labels = torch.randint(0, 5, (8,))
        loss = MulticlassHingeWithMargin(margin=0.5)(logits, labels)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_max_not_sum_over_wrong_classes(self):
        # Distinguishes Tsuzuku (max) from PyTorch's standard hinge (sum).
        # Two wrong classes both violate the margin: a sum-loss would
        # accumulate both, the max-loss returns only the worst.
        logits = torch.tensor([[0., 4., 4.]])
        labels = torch.tensor([0])
        loss = MulticlassHingeWithMargin(margin=1.0)(logits, labels)
        # max(0, 1 + 4 - 0) = 5, taken over the worst (tied) wrong class.
        self.assertAlmostEqual(loss.item(), 5.0, places=6)


if __name__ == '__main__':
    unittest.main()
