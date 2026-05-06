from typing import Callable, Optional, Protocol

from torch.utils.data import DataLoader, Dataset, random_split


def get_data_loader(trainset: Dataset,
                    valset: Optional[Dataset],
                    batch_size: int,
                    num_workers: int,
                    log: Callable[[str], None],  # e.g. logging.info
                    ):
    if valset is None:
        # No held-out valset given: 90/10 split of trainset.
        log('No valset provided; using 90/10 split of trainset.')
        len_trainset = int(len(trainset) * 0.9)
        trainset, valset = random_split(
            trainset, [len_trainset, len(trainset) - len_trainset])

    log(f'Training dataset: {trainset}')
    log(f'Validation dataset: {valset}')
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(dataset=valset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


class DatasetClass(Protocol):
    def __call__(self, train: bool) -> Dataset:
        ...


def get_train_val_loader(dataset_cls: DatasetClass,
                         batch_size: int = 256,
                         num_workers: int = 4,
                         log: Callable[[str], None] = print,
                         ):
    train_dataset = dataset_cls(train=True)
    train_loader, test_loader = get_data_loader(
        train_dataset,
        valset=None,
        batch_size=batch_size,
        num_workers=num_workers,
        log=log,
    )
    return train_loader, test_loader


def get_train_test_loader(dataset_cls: DatasetClass,
                          batch_size: int = 256,
                          num_workers: int = 4,
                          log: Callable[[str], None] = print,
                          ):
    train_dataset = dataset_cls(train=True)
    test_dataset = dataset_cls(train=False)
    train_loader, test_loader = get_data_loader(
        train_dataset,
        valset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        log=log,
    )
    return train_loader, test_loader
