#!/usr/bin/env python3
"""Drift of HH/LS/NA activation parameters vs their inits, on FashionMNIST checkpoints.

LS drift is reported on `coef.original` (pre-projection); since SplineProj is
mean-preserving, this upper-bounds the actual function-shape change. NA inits
are pre-divided by lr_scale so absid is (0,0)/(odd: -1000,0) in `theta_raw` units.
"""
import math
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(REPO_ROOT, 'results/FashionMNIST')
OUT_DIR = os.path.join(REPO_ROOT, 'docs/papers/figures')

LAYERS = ['AOL', 'SOC', 'CPL']
ACTIVATIONS = ['HH', 'LS', 'NA']  # stateless ones (GS, AV) skipped


def hh_drift(state):
    """Return (drifts, init_value) for all HH theta tensors."""
    drifts = []
    for k, v in state.items():
        if k.endswith('.theta') and v.dim() == 1:
            init = torch.full_like(v, math.pi / 2)
            drifts.append((k, v - init))
    return drifts, math.pi / 2


def ls_drift(state):
    """Return (drifts, init_label) for all LS coef.original tensors."""
    drifts = []
    init_label = '|x_k|, x in linspace(-3,3,21)'
    for k, v in state.items():
        if k.endswith('.parametrizations.coef.original') and v.dim() == 2:
            num_knots = v.shape[1]
            xs = torch.linspace(-3.0, 3.0, num_knots)
            init = xs.abs().unsqueeze(0).expand_as(v).clone()
            drifts.append((k, v - init))
    return drifts, init_label


def na_drift(state):
    """Return ([(name, drift)], init_label) for NActivation theta1_raw/theta2_raw."""
    drifts = []
    init_label = 'absid: identity even-ch, abs odd-ch (theta1_raw odd = -1000)'
    for k, v in state.items():
        if k.endswith('.theta1_raw'):
            init = torch.zeros_like(v)
            init[1::2] = -1000.0
            drifts.append((k, v - init))
        elif k.endswith('.theta2_raw'):
            init = torch.zeros_like(v)
            drifts.append((k, v - init))
    return drifts, init_label


DRIFT_FNS = {
    'HH': hh_drift,
    'LS': ls_drift,
    'NA': na_drift,
}


def summarize_drift(drifts):
    """Aggregate per-element drift across all activation modules in the network."""
    if not drifts:
        return None
    all_d = torch.cat([d.flatten().abs() for _, d in drifts])
    signed = torch.cat([d.flatten() for _, d in drifts])
    return {
        'n_params': all_d.numel(),
        'n_modules': len(drifts),
        'mean_abs': all_d.mean().item(),
        'std_abs': all_d.std().item(),
        'max_abs': all_d.max().item(),
        'min_signed': signed.min().item(),
        'max_signed': signed.max().item(),
        'rms': signed.pow(2).mean().sqrt().item(),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, 'fmnist_param_drift.txt')

    lines = []
    header = (f'{"cell":<18}  {"#params":>10}  {"#modules":>9}  '
              f'{"mean|drift|":>11}  {"std|drift|":>11}  '
              f'{"max|drift|":>11}  {"min(signed)":>11}  {"max(signed)":>11}  '
              f'{"RMS":>9}')
    lines.append(header)
    lines.append('-' * len(header))

    for layer in LAYERS:
        for act in ACTIVATIONS:
            cell = f'XS_{layer}_{act}_CE'
            pth = os.path.join(RESULTS_DIR, f'{cell}.pth')
            if not os.path.isfile(pth):
                lines.append(f'{cell:<18}  MISSING ({pth})')
                continue
            state = torch.load(pth, map_location='cpu', weights_only=True)
            drifts, _init_label = DRIFT_FNS[act](state)
            s = summarize_drift(drifts)
            if s is None:
                lines.append(f'{cell:<18}  no activation params found')
                continue
            lines.append(
                f'{cell:<18}  '
                f'{s["n_params"]:>10d}  '
                f'{s["n_modules"]:>9d}  '
                f'{s["mean_abs"]:>11.4f}  '
                f'{s["std_abs"]:>11.4f}  '
                f'{s["max_abs"]:>11.4f}  '
                f'{s["min_signed"]:>11.4f}  '
                f'{s["max_signed"]:>11.4f}  '
                f'{s["rms"]:>9.4f}')

    lines.append('')
    lines.append('Init values:')
    lines.append('  HH theta:        init = pi/2 (~1.5708) per channel-pair')
    lines.append('  LS coef.original (init=absolute_value):')
    lines.append('                   init[k] = |x_k|, x_k in linspace(-3, 3, 21);')
    lines.append('                   knot heights at +/- 3,2.7,...,0.3,0,0.3,...,2.7,3')
    lines.append('  NA theta1_raw:   init=absid -> 0 for even channels, -1000 for odd')
    lines.append('  NA theta2_raw:   init=absid -> 0 for all channels')
    lines.append('')
    lines.append(
        'Note: LS drift is reported in coef.original (the SGD-learned tensor)'
        ' rather than in SplineProj(coef.original). Since |x| already satisfies'
        ' the 1-Lipschitz constraint, projection at init is a no-op, so this'
        ' is a faithful measure of how much the spline shape moved.')

    text = '\n'.join(lines) + '\n'
    with open(out_path, 'w') as f:
        f.write(text)
    print(text)
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
