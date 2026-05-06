
from torch import nn

from .absolute_value import AbsoluteValue as Abs
from .householder import Householder
from .learnable_linear_spline import LinearSpline
from .max_min import MaxMin
from .n_activation import NActivation


class Identity(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    @staticmethod
    def forward(x):
        return x
