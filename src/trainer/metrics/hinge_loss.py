import torch
from torch import Tensor

from .base_metric_class import Metric


class MulticlassHingeWithMargin(Metric):
    """Tsuzuku et al. (2018) multiclass hinge: max(0, margin + max_{j!=y} f_j - f_y).

    `margin` is in raw-logit units; no division by sqrt(2) or by the network
    Lipschitz constant since the network is 1-Lipschitz by construction.
    """

    def __init__(self, margin: float = 0.0) -> None:
        super().__init__('mean')
        self.margin = margin

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        true_logit = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        mask = torch.zeros_like(logits).scatter_(
            1, labels.unsqueeze(1), float('-inf'))
        best_other = (logits + mask).max(dim=1).values
        return torch.clamp_min(self.margin + best_other - true_logit, 0.).mean()
