#!/usr/bin/env python3
"""val_loss / val_accuracy / val_robstacc_36 vs epoch for the 15-cell FashionMNIST grid."""
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS_BASE = os.path.join(REPO_ROOT, 'data/runs/FashionMNIST')
OUT_DIR = os.path.join(REPO_ROOT, 'docs/papers/figures')

LAYERS = ['AOLConv2d', 'SOC', 'CPLConv2d']
ACTIVATIONS = ['GS', 'AV', 'HH', 'LS', 'NA']
ACT_LABELS = {
    'GS': 'MaxMin',
    'AV': 'AbsoluteValue',
    'HH': 'Householder',
    'LS': 'LinearSpline',
    'NA': 'NActivation',
}
ACT_COLORS = {
    'GS': 'tab:blue',
    'AV': 'tab:orange',
    'HH': 'tab:green',
    'LS': 'tab:red',
    'NA': 'tab:purple',
}


def find_csv(act: str, layer: str) -> str:
    pattern = os.path.join(
        RUNS_BASE, f'grid_final_{act}', 'ConvNetXS', layer,
        '2026*', 'training_statistics.csv')
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    # If multiple timestamps exist, take the latest.
    return matches[-1]


def plot_layer(layer: str, fig_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ('val_loss', 'val loss', 'log'),
        ('val_accuracy', 'val accuracy', 'linear'),
        ('val_robstacc_36', 'val robust acc @ eps=36/255 (CRA)', 'linear'),
    ]
    n_loaded = 0
    for act in ACTIVATIONS:
        csv_path = find_csv(act, layer)
        if csv_path is None:
            print(f'  WARN: no csv for {act}/{layer}', file=sys.stderr)
            continue
        df = pd.read_csv(csv_path)
        n_loaded += 1
        for ax, (col, ylabel, scale) in zip(axes, metrics):
            if col not in df.columns:
                continue
            ax.plot(df['epoch'], df[col],
                    color=ACT_COLORS[act], label=ACT_LABELS[act],
                    linewidth=1.2, alpha=0.85)
            ax.set_xlabel('epoch')
            ax.set_ylabel(ylabel)
            if scale == 'log':
                ax.set_yscale('log')
            ax.grid(alpha=0.3)
    axes[0].legend(loc='upper right', fontsize=9, framealpha=0.85)
    fig.suptitle(f'ConvNetXS / {layer.replace("Conv2d", "")} / FashionMNIST '
                 f'(LipCE) — {n_loaded}/5 activations',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  Wrote {fig_path}')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for layer in LAYERS:
        out = os.path.join(OUT_DIR,
                           f'fmnist_convergence_{layer.replace("Conv2d", "")}.png')
        print(f'Plotting {layer}...')
        plot_layer(layer, out)
    print('Done.')


if __name__ == '__main__':
    main()
