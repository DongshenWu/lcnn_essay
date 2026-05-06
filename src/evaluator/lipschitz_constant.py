import torch

from models.layers.utils import module_symmetric_power_iteration_all_iterations


@torch.no_grad()
def bound_lipschitz_constant_all_iterations(model,
                                            model_input_size: int = 32,
                                            nrof_iterations=1000):
    """Per-layer power iteration; returns the running product across iterations.

    Use ~1e3 iters for per-layer estimates, ~1e5 for whole-model bounds.
    """
    all_iterations = None
    modules = [m for m in model.modules() if not isinstance(m, torch.nn.Sequential)]
    # In our ConvNet blocks, side_length * channels is invariant across PixelUnshuffle.
    product = None

    exceptions = ['SOC', 'BCOP', 'ECO']
    for module in modules:
        model_is_linear = isinstance(module, torch.nn.Linear)
        model_is_conv = isinstance(module, torch.nn.Conv2d)
        model_is_exception = any(e in module.__class__.__name__ for e in exceptions)

        if not (model_is_conv or model_is_linear or model_is_exception):
            continue

        nrof_channels = (module.in_channels if model_is_conv or model_is_exception
                         else module.in_features)
        if product is None:
            product = nrof_channels * model_input_size
        side_length = product // nrof_channels

        input_size = ((nrof_channels, side_length, side_length)
                      if model_is_conv or model_is_exception
                      else (nrof_channels,))

        layer_all_iterations = module_symmetric_power_iteration_all_iterations(
            module, input_size, nrof_iterations=nrof_iterations)

        if all_iterations is None:
            all_iterations = layer_all_iterations
        else:
            all_iterations = all_iterations * layer_all_iterations

    return all_iterations


def bound_lipschitz_constant(*args, **kwargs):
    bound_all_iterations = bound_lipschitz_constant_all_iterations(
        *args, **kwargs)
    if bound_all_iterations is None:
        return torch.tensor(1.)
    else:
        return bound_all_iterations[-1]
