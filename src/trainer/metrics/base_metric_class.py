
from torch import nn, Tensor
from typing import Union, Callable
import torch


class Metric(nn.Module):
    """Base class for YAML-selectable eval metrics. `aggregation` is the cross-batch reduction
    used by Statistics; the per-batch loss value passes through unaggregated.
    """

    def __init__(self, aggregation: Union[str, Callable] = 'mean') -> None:
        self.aggregation = aggregation
        super().__init__()

    def forward(self, out: Tensor, labels: Tensor) -> Tensor:
        raise NotImplementedError

    def get_name(self):
        if hasattr(self, "name"):
            return self.name
        return self.__class__.__name__

    def aggregated(self, *args, **kwargs):
        """ Batch aggreated. (pytorch would call it reduced) """
        pre_aggregated = self(*args, **kwargs)
        if self.aggregation == 'none':
            return pre_aggregated

        return getattr(torch, self.aggregation)(pre_aggregated.float())
