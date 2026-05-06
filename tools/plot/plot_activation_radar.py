#!/usr/bin/env python3
"""6D radar plots (A / TT / IT / TM / IM / RA) per activation, one figure per
normalisation (min-max, rank). Reads results/<DATASET>/XS_<LAYER>_<ACT>_CE.txt.
"""
import argparse
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import rankdata

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASETS = ['CIFAR10', 'CIFAR100', 'FashionMNIST']
LAYERS = ['AOL', 'CPL', 'SOC']
ACTIVATIONS = ['AV', 'GS', 'HH', 'LS', 'NA']
ACT_FULL = {
    'AV': 'AbsoluteValue',
    'GS': 'MaxMin',
    'HH': 'Householder',
    'LS': 'LinearSpline',
    'NA': 'NActivation',
}

EPS_VALUES = np.array([0.1411, 0.2824, 0.4235, 1.0000])
EPS_RANGE = EPS_VALUES[-1] - EPS_VALUES[0]

AXES = ['CA', 'TT', 'IT', 'TM', 'IM', 'RA']
AXIS_FULL = {
    'CA': 'Clean Accuracy',
    'TT': 'Train Time / Epoch',
    'IT': 'Inference Throughput',
    'TM': 'Train Memory',
    'IM': 'Inference Memory',
    'RA': 'Robust Accuracy',
}
HIGHER_BETTER = {'CA': True, 'RA': True, 'IT': True,
                 'TT': False, 'TM': False, 'IM': False}

ACT_FILL = {
    'AV': '#9ec5e8',
    'GS': '#f7c98a',
    'HH': '#b2dfb2',
    'LS': '#e6a3a3',
    'NA': '#c3b1d8',
}
ACT_EDGE = {
    'AV': '#3d7fbf',
    'GS': '#d68a2c',
    'HH': '#4ea64e',
    'LS': '#b94a4a',
    'NA': '#7a5fa8',
}

CRA_LINE = re.compile(
    r'^\s*([0-9.]+)\s+(\d+)/255\s+([0-9.]+)%\s+([0-9.]+)%\s+([+-]?[0-9.]+)pp\s*$'
)
CLEAN_ACC = re.compile(r'clean accuracy\s*=\s*([\d.]+)%')
EPOCHS = re.compile(r'epochs\s*=\s*(\d+)')
WALL_CLOCK = re.compile(r'wall_clock\s*=\s*[^(]+\(([\d.]+)\s*s')
TRAIN_MEM = re.compile(r'train\s*\(forward\+backward,\s*batch=\d+\)\s*=\s*([\d.]+)')
EVAL_MEM = re.compile(r'eval\s*\(forward only,\s*batch=\d+\)\s*=\s*([\d.]+)')
EVAL_THROUGHPUT = re.compile(r'^\s*eval\s*=\s*([\d.]+)\s*$', re.MULTILINE)


def parse_archive(path):
    with open(path) as f:
        text = f.read()

    cra_rows = {}
    for line in text.splitlines():
        m = CRA_LINE.match(line)
        if m:
            cra_rows[int(m.group(2))] = float(m.group(3))
    expected = {36, 72, 108, 255}
    if set(cra_rows) != expected:
        raise ValueError(f'{path}: expected CRA rows at {expected}, got {set(cra_rows)}')
    cra_pct = np.array([cra_rows[r] for r in (36, 72, 108, 255)])
    auc_cra = float(np.trapz(cra_pct, EPS_VALUES) / (EPS_RANGE * 100.0))

    def grab(rx, label):
        m = rx.search(text)
        if m is None:
            raise ValueError(f'{path}: could not parse {label}')
        return float(m.group(1))

    clean = grab(CLEAN_ACC, 'clean accuracy') / 100.0
    epochs = int(grab(EPOCHS, 'epochs'))
    wall = grab(WALL_CLOCK, 'wall_clock')
    train_mem = grab(TRAIN_MEM, 'train memory')
    eval_mem = grab(EVAL_MEM, 'eval memory')
    eval_tp = grab(EVAL_THROUGHPUT, 'eval throughput')

    return {
        'CA': clean,
        'RA': auc_cra,
        'TT': wall / epochs,
        'IT': eval_tp,
        'TM': train_mem,
        'IM': eval_mem,
    }


def load_all(results_dir):
    data = {}
    for ds in DATASETS:
        for layer in LAYERS:
            for act in ACTIVATIONS:
                path = os.path.join(results_dir, ds, f'XS_{layer}_{act}_CE.txt')
                if not os.path.exists(path):
                    raise FileNotFoundError(f'missing required cell: {path}')
                data[(ds, layer, act)] = parse_archive(path)
    return data


def compress_minmax(per_cell_values):
    """per_cell_values: array of shape (n_acts,). Returns higher-is-better in [0,1]."""
    vmin = per_cell_values.min()
    vmax = per_cell_values.max()
    if vmax == vmin:
        return np.ones_like(per_cell_values)
    return (per_cell_values - vmin) / (vmax - vmin)


def compress_rank(per_cell_values):
    """per_cell_values: array of shape (n_acts,). Returns higher-is-better in [0,1].

    Higher-better convention: largest raw value gets rank 1 (best), so we negate
    before ranking. rankdata returns 1 = smallest by default.
    """
    n = len(per_cell_values)
    ranks = rankdata(-per_cell_values, method='average')
    return 1.0 - (ranks - 1.0) / (n - 1)


def compress_blend(per_cell_values, alpha=0.5):
    """Convex combination of min-max and rank scores: alpha*minmax + (1-alpha)*rank.

    Rank-preserving (both inputs are monotone in raw value); continuous in raw
    value (min-max term slides each value within its rank-bucket); magnitude-
    aware (gap size shows up via the min-max term); damped against single-cell
    outliers (rank term anchors to a discrete grid regardless of spread).
    """
    return alpha * compress_minmax(per_cell_values) + (1.0 - alpha) * compress_rank(per_cell_values)


def compute_scores(data, method):
    """Return scores[axis][activation] = mean of per-cell normalised values.

    method in {'minmax', 'rank', 'blend'}.
    """
    compressor = {
        'minmax': compress_minmax,
        'rank':   compress_rank,
        'blend':  compress_blend,
    }[method]

    scores = {axis: {act: [] for act in ACTIVATIONS} for axis in AXES}
    for ds in DATASETS:
        for layer in LAYERS:
            for axis in AXES:
                raw = np.array([data[(ds, layer, act)][axis] for act in ACTIVATIONS])
                if not HIGHER_BETTER[axis]:
                    raw = -raw
                norm = compressor(raw)
                for act, v in zip(ACTIVATIONS, norm):
                    scores[axis][act].append(float(v))

    # Verify each cell contributed, then collapse to mean.
    n_cells = len(DATASETS) * len(LAYERS)
    aggregated = {axis: {} for axis in AXES}
    for axis in AXES:
        for act in ACTIVATIONS:
            vals = scores[axis][act]
            assert len(vals) == n_cells, f'{axis} {act}: got {len(vals)} cells'
            assert all(0.0 - 1e-9 <= v <= 1.0 + 1e-9 for v in vals), \
                f'{axis} {act}: out-of-bounds {vals}'
            aggregated[axis][act] = float(np.mean(vals))
    return aggregated


def _draw_polar(ax, scores, act, theta, closed_theta):
    """Render one activation's polygon onto a polar axis."""
    values = np.array([scores[a][act] for a in AXES])
    closed_vals = np.concatenate([values, values[:1]])

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Floor-at-1: rescale s in [0,1] -> r in [1,5] so worst-on-every-cell
    # still has visible base pentagon at r=1 (matches reference image scale).
    radii = 1.0 + 4.0 * closed_vals

    ax.fill(closed_theta, radii, color=ACT_FILL[act], alpha=0.4, linewidth=0)
    ax.plot(closed_theta, radii, color=ACT_EDGE[act], linewidth=1.8)

    ax.set_xticks(theta)
    ax.set_xticklabels(AXES, fontsize=10)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels([])
    ax.grid(color='0.85', linewidth=0.8)
    ax.spines['polar'].set_color('0.7')

    ax.text(0.5, 0.5, act, ha='center', va='center',
            fontsize=12, fontweight='bold', transform=ax.transAxes)


def _draw_legend(ax_leg):
    """Render the abbreviation legend on a non-polar axis."""
    ax_leg.axis('off')
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    ax_leg.text(0.5, 0.92, 'Legend', ha='center', va='top',
                fontsize=12, fontweight='bold',
                transform=ax_leg.transAxes)
    box = plt.Rectangle((0.10, 0.12), 0.80, 0.74,
                        fill=False, edgecolor='0.6', linewidth=0.8,
                        transform=ax_leg.transAxes)
    ax_leg.add_patch(box)
    y0 = 0.78
    dy = 0.11
    for i, axis in enumerate(AXES):
        y = y0 - i * dy
        ax_leg.text(0.18, y, axis, ha='left', va='center',
                    fontsize=11, fontweight='bold',
                    transform=ax_leg.transAxes)
        ax_leg.text(0.82, y, AXIS_FULL[axis], ha='right', va='center',
                    fontsize=11, fontstyle='italic',
                    transform=ax_leg.transAxes)


def plot_radar(scores, method_label, png_path, pdf_path):
    """1x5 row layout, no legend."""
    n_axes = len(AXES)
    # Reference image places CA at 1 o'clock and goes clockwise: CA, TT, IT, TM, IM, RA.
    # We set theta_zero='N' (theta=0 -> 12 o'clock) and theta_direction=-1 (clockwise),
    # then place axes at angles offset by +pi/6 so CA lands at 1 o'clock.
    theta = np.linspace(0, 2 * np.pi, n_axes, endpoint=False) + np.pi / 6
    closed_theta = np.concatenate([theta, theta[:1]])

    fig = plt.figure(figsize=(18, 4.5))
    for i, act in enumerate(ACTIVATIONS):
        ax = fig.add_subplot(1, 5, i + 1, projection='polar')
        _draw_polar(ax, scores, act, theta, closed_theta)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Wrote {png_path}')
    print(f'  Wrote {pdf_path}')


def plot_radar_grid(scores, png_path):
    """2x3 grid layout: 5 activation polars + legend in the 6th cell."""
    n_axes = len(AXES)
    theta = np.linspace(0, 2 * np.pi, n_axes, endpoint=False) + np.pi / 6
    closed_theta = np.concatenate([theta, theta[:1]])

    fig = plt.figure(figsize=(12, 8))
    for i, act in enumerate(ACTIVATIONS):
        ax = fig.add_subplot(2, 3, i + 1, projection='polar')
        _draw_polar(ax, scores, act, theta, closed_theta)

    ax_leg = fig.add_subplot(2, 3, 6)
    _draw_legend(ax_leg)

    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Wrote {png_path}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results-dir', default=os.path.join(REPO_ROOT, 'results'))
    p.add_argument('--output-dir', default=os.path.join(REPO_ROOT, 'results', 'figures'))
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Loading {len(DATASETS) * len(LAYERS) * len(ACTIVATIONS)} cells '
          f'from {args.results_dir}')
    data = load_all(args.results_dir)

    for method, label, stem in [
        ('blend', 'per-cell 0.5*min-max + 0.5*rank, mean over 9 cells',
         'activation_radar'),
    ]:
        print(f'Compression: {method}')
        scores = compute_scores(data, method)
        for axis in AXES:
            row = '  '.join(f'{act}={scores[axis][act]:.3f}' for act in ACTIVATIONS)
            print(f'    {axis}: {row}')
        png = os.path.join(args.output_dir, f'{stem}.png')
        pdf = os.path.join(args.output_dir, f'{stem}.pdf')
        plot_radar(scores, label, png, pdf)
        png_grid = os.path.join(args.output_dir, f'{stem}_grid.png')
        plot_radar_grid(scores, png_grid)


if __name__ == '__main__':
    main()
