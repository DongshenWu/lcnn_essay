"""Fashion-MNIST loader. Pads 28x28 to 32x32 so the upstream ConvNet blocks (5x PixelUnshuffle(2)) work unchanged."""
from typing import Optional

from torchvision import datasets
from torchvision import transforms as tsfm
from torchvision.transforms import InterpolationMode as Interp

DATA_ROOT = 'data/datasets'

FashionMNIST_mean = [0.2860]
FashionMNIST_std = [0.3530]


class FashionMNIST(datasets.FashionMNIST):
    def __init__(self, train=True, center=True, rescale=False,
                 size: Optional[int] = None) -> None:
        transf_list = [tsfm.ToTensor(), tsfm.Pad(2)]
        if center:
            transf_list.append(tsfm.Normalize(FashionMNIST_mean, [1.]))
        if rescale:
            transf_list.append(tsfm.Normalize(FashionMNIST_mean, FashionMNIST_std))
        if train:
            transf_list.append(tsfm.RandomCrop(32, 4))
        if size is not None:
            transf_list.append(tsfm.Resize(size, Interp.NEAREST, antialias=None))

        super().__init__(
            root=DATA_ROOT, train=train,
            transform=tsfm.Compose(transf_list), download=True,
        )
