"""L2 PGD empirical-robustness evaluator (Madry et al., 2018).

Companion to certified-RA at `evaluator/robust_accuracy.py`; the gap
emp_RA - CRA quantifies how loose each Lipschitz parameterisation's certificate is.
Attack runs in centered input space (matching CRA), so we drop the [0,1] pixel
clamp that `torchattacks.PGDL2` applies — those bounds don't correspond to pixel
validity once the data is centered, and CRA certifies all L2-bounded perturbations.
"""
from typing import Iterable, List, Optional

import logging

import numpy
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

import torchattacks

from trainer.metrics import Margin

SQRT2 = 2 ** 0.5


class L2PGDAttack(torchattacks.PGDL2):
    """`torchattacks.PGDL2` with the [0,1] pixel clamps removed."""

    def forward(self, images: Tensor, labels: Tensor) -> Tensor:
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        loss = nn.CrossEntropyLoss()
        batch_size = images.shape[0]
        adv_images = images.clone().detach()

        if self.random_start:
            # Sample a uniform point in the L2 ball of radius eps:
            #   direction ~ N(0, I) / ||.||,  radius ~ U(0, 1) * eps
            # (equivalent to the "uniform on the sphere x uniform radius" trick.)
            delta = torch.empty_like(adv_images).normal_()
            d_flat = delta.view(batch_size, -1)
            n = d_flat.norm(p=2, dim=1).view(batch_size, *([1] * (delta.ndim - 1)))
            r = torch.zeros_like(n).uniform_(0, 1)
            delta = delta * r / (n + self.eps_for_division) * self.eps
            adv_images = (adv_images + delta).detach()

        for _ in range(self.steps):
            # Forward + backward to get the loss gradient w.r.t. the adversarial image.
            # retain_graph/create_graph=False: we only need a first-order gradient.
            adv_images.requires_grad_(True)
            outputs = self.get_logits(adv_images)
            cost = loss(outputs, labels)
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False)[0]

            # Madry-style normalised step: move alpha along the L2-unit gradient direction.
            # Per-sample normalisation keeps the step size eps-comparable across the batch
            # regardless of the gradient magnitude.
            grad_norms = (
                grad.view(batch_size, -1).norm(p=2, dim=1)
                + self.eps_for_division
            ).view(batch_size, *([1] * (grad.ndim - 1)))
            grad = grad / grad_norms
            adv_images = adv_images.detach() + self.alpha * grad

            # Project back into the L2 eps-ball around the original image. factor < 1 only
            # when ||delta||_2 > eps; otherwise the step stayed inside the ball and we
            # leave it untouched. Note: the upstream torchattacks.PGDL2 also clamps adv to
            # [0, 1] here — we removed that, since our inputs are mean-centered tensors
            # (range ~[-mean, 1-mean] per channel), so [0, 1] no longer means "valid pixel"
            # and clipping there would shrink the attack budget below CRA's threat model.
            delta = adv_images - images
            delta_norms = delta.view(batch_size, -1).norm(p=2, dim=1)
            factor = torch.clamp_max(
                self.eps / (delta_norms + self.eps_for_division), 1.0)
            delta = delta * factor.view(batch_size, *([1] * (delta.ndim - 1)))
            adv_images = (images + delta).detach()

        return adv_images


class EvaluatePGD:
    """Per-batch attack at every eps; logs both CRA and emp-RA so soundness (CRA <= emp_RA) is self-checking."""

    def __init__(
        self,
        model: Module,
        valset: Dataset,
        device: torch.device,
        eps_list: Iterable[float],
        n_iter: int = 20,
        step_size: Optional[float] = None,
        random_start: bool = True,
        batch_size: int = 100,
        num_workers: int = 2,
        logger=logging,
    ) -> None:
        self.logger = logger
        self.device = device
        self.model = model.to(device).eval()
        self.test_loader = DataLoader(
            valset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers,
        )
        self.eps_list: List[float] = list(eps_list)
        self.n_iter = n_iter
        self.step_size = step_size
        self.random_start = random_start

        # One attack instance per eps. Default alpha = 2.5 * eps / n_iter (Madry):
        # 2.5x the per-step distance needed to traverse the ball lets the attack reach
        # the boundary even when the random start is on the opposite side.
        self._attacks = {
            eps: L2PGDAttack(
                self.model,
                eps=eps,
                alpha=(2.5 * eps / n_iter) if step_size is None else step_size,
                steps=n_iter,
                random_start=random_start,
            )
            for eps in self.eps_list
        }
        self._margin = Margin()

        self.metrics: dict[str, list] = {"label": [], "clean_correct": []}
        for eps in self.eps_list:
            self.metrics[f"cra_{eps:.4f}"] = []
            self.metrics[f"emp_ra_{eps:.4f}"] = []

    def evaluate_batch(self, x: Tensor, y: Tensor) -> None:
        # Clean forward + certified margin in a single no-grad pass. Margin / sqrt(2) is
        # the L2 distance to the decision boundary for a 1-Lipschitz network: the model's
        # Lipschitz constant in input->logit is 1, so any L2 perturbation strictly smaller
        # than margin/sqrt(2) cannot flip the argmax.
        with torch.no_grad():
            clean_logits = self.model(x)
            margin = self._margin(clean_logits, y) / SQRT2
            clean_pred = clean_logits.argmax(-1).long()
        self.metrics["label"].append(self._to_numpy(y))
        self.metrics["clean_correct"].append(
            self._to_numpy((clean_pred == y).float()))

        for eps in self.eps_list:
            # CRA(eps): correct AND certified-robust at radius eps. The clean_pred==y guard
            # is needed because Margin clamps to 0 on misclassification, which would
            # otherwise satisfy `margin > eps` only at eps<0.
            self.metrics[f"cra_{eps:.4f}"].append(
                self._to_numpy(((margin > eps) & (clean_pred == y)).float()))
            # emp_RA(eps): run PGD at this eps, count samples still classified correctly.
            # Soundness invariant for a 1-Lipschitz layer: CRA(eps) <= emp_RA(eps) <= clean
            # (PGD cannot beat a valid certificate). A violation indicates the layer is not
            # actually 1-Lipschitz at the asserted radius — emp_RA is then the trustworthy
            # number, and the gap is essay-relevant evidence about the certificate's looseness.
            x_adv = self._attacks[eps](x, y)
            with torch.no_grad():
                adv_pred = self.model(x_adv).argmax(-1).long()
            self.metrics[f"emp_ra_{eps:.4f}"].append(
                self._to_numpy((adv_pred == y).float()))

    def run(self, save_path: str | None) -> dict:
        n_batches = len(self.test_loader)
        for idx, (x, y) in enumerate(self.test_loader, 1):
            x = x.to(self.device)
            y = y.to(self.device)
            self.evaluate_batch(x, y)
            if idx % 10 == 0 or idx == n_batches:
                self.logger.info(f"PGD batch {idx}/{n_batches}")
        if save_path is not None:
            self.save(save_path)
        return self.summary()

    def save(self, save_path: str) -> None:
        flat = {k: numpy.concatenate(v) for k, v in self.metrics.items()}
        df = pd.DataFrame(flat)
        df.to_csv(save_path, index=False)
        self.logger.info(f"Wrote {save_path} ({len(df)} samples)")

    def summary(self) -> dict:
        out = {
            "clean_acc": float(
                numpy.concatenate(self.metrics["clean_correct"]).mean()),
        }
        for eps in self.eps_list:
            out[f"cra_{eps:.4f}"] = float(
                numpy.concatenate(self.metrics[f"cra_{eps:.4f}"]).mean())
            out[f"emp_ra_{eps:.4f}"] = float(
                numpy.concatenate(self.metrics[f"emp_ra_{eps:.4f}"]).mean())
        return out

    @staticmethod
    def _to_numpy(t: Tensor) -> numpy.ndarray:
        return t.detach().cpu().numpy()
