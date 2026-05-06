from typing import Callable

import torch


def add_unit(nrof_bytes: int):
    if nrof_bytes > 10**9:
        return f"{nrof_bytes / 10**9:.2f} GB"
    if nrof_bytes > 10**6:
        return f"{nrof_bytes / 10**6:.2f} MB"
    if nrof_bytes > 10**3:
        return f"{nrof_bytes / 10**3:.2f} KB"
    return f"{nrof_bytes} Bytes"


def print_memory(description=None):
    torch.cuda.empty_cache()
    print("="*40)
    if description is not None:
        print(description + ":")
    print("Memory reserved:", add_unit(torch.cuda.memory_reserved()))
    print("Max memory reserved:", add_unit(torch.cuda.max_memory_reserved()))
    print("Memory allocated:", add_unit(torch.cuda.memory_allocated()))
    print("Max memory allocated:", add_unit(torch.cuda.max_memory_allocated()))
    print("="*40)


def get_max_memory_usage(target_function, *args, **kwargs):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    pre_allocated = torch.cuda.max_memory_allocated()
    if pre_allocated > 0:
        print(f"\nWarning: Cuda memory is not empty!")
        print_memory("Current cuda memory")

    target_function(*args, **kwargs)
    return torch.cuda.max_memory_allocated() - pre_allocated


@torch.no_grad()
@torch.nn.utils.parametrize.cached()
def forward_passes(model, data_loader, device="cuda"):
    model.to(device)
    model.eval()
    for i, (batch, _) in enumerate(data_loader):
        batch = batch.to(device)
        out = model(batch)
        out.sum()
        del batch
        del out
        # >=2 passes so e.g. SGD momentum buffers are accounted for in peak memory.
        if i >= 2:
            break


def backward_passes(model, data_loader, optimizer, device="cuda"):
    model.to(device)
    model.train()
    for i, (batch, _) in enumerate(data_loader):
        batch = batch.to(device)
        out = model(batch)
        out.sum().backward()
        optimizer.step()
        del batch
        del out
        # >=2 batches so optimizer.step() momentum tensors are allocated and counted.
        if i >= 2:
            break


def get_model_memory(get_model: Callable[[], torch.nn.Module],
                     test_loader,
                     train_loader,
                     logging=print,
                     ):

    if logging is None:
        def logging(*args, **kwargs): pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "Only CUDA is supported for now."
    logging(f"Using device \"{device}\".")

    test_model = get_model()
    def forward():
        forward_passes(test_model, test_loader, device=device)
    test_max_memory = get_max_memory_usage(target_function=forward)
    logging(f"Test max memory: {add_unit(test_max_memory)}.")
    del forward
    del test_model

    train_model = get_model()
    train_mom_sgd = torch.optim.SGD(train_model.parameters(), lr=0.01, momentum=0.9)
    def backward():
        backward_passes(train_model, train_loader, train_mom_sgd, device=device)

    train_max_memory = get_max_memory_usage(target_function=backward)
    logging(f"Train max memory: {add_unit(train_max_memory)}.")

    return {
        "max_train_memory": train_max_memory,
        "max_test_memory": test_max_memory,
    }

