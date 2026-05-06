#!/usr/bin/env python3
"""2x3 grid: CRA (top) and emp-RA - CRA gap (bottom) vs eps, per (layer, activation)."""
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARCHIVE_DIR = os.path.join(REPO_ROOT, 'results', 'CIFAR10')
OUT_DIR = os.path.join(REPO_ROOT, 'docs', 'papers', 'figures')

LAYERS = ['AOL', 'CPL', 'SOC']
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

RADII = [36, 72, 108, 255]
RADII_LABELS = [f'{r}/255' for r in RADII]

DATA_LINE = re.compile(
    r'^\s*([0-9.]+)\s+(\d+)/255\s+([0-9.]+)%\s+([0-9.]+)%\s+([+-]?[0-9.]+)pp\s*$'
)


def parse_archive(path):
    """Return ([cra...], [gap...]) ordered to match RADII, or None on failure."""
    rows = {}
    with open(path) as f:
        for line in f:
            m = DATA_LINE.match(line)
            if m:
                radius = int(m.group(2))
                cra = float(m.group(3))
                gap = float(m.group(5))
                rows[radius] = (cra, gap)
    if set(rows) != set(RADII):
        return None
    cra = [rows[r][0] for r in RADII]
    gap = [rows[r][1] for r in RADII]
    return cra, gap


def load_all():
    data = {}
    for layer in LAYERS:
        for act in ACTIVATIONS:
            path = os.path.join(ARCHIVE_DIR, f'XS_{layer}_{act}_CE.txt')
            if not os.path.exists(path):
                print(f'  WARN: missing {path}', file=sys.stderr)
                continue
            parsed = parse_archive(path)
            if parsed is None:
                print(f'  WARN: could not parse 4 radii from {path}',
                      file=sys.stderr)
                continue
            data[(layer, act)] = parsed
    return data


def plot(data, png_path, pdf_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8),
                             sharex='col', sharey='row')

    for col, layer in enumerate(LAYERS):
        ax_cra = axes[0, col]
        ax_gap = axes[1, col]
        for act in ACTIVATIONS:
            if (layer, act) not in data:
                continue
            cra, gap = data[(layer, act)]
            ax_cra.plot(RADII, cra, color=ACT_COLORS[act],
                        label=ACT_LABELS[act], marker='o',
                        linewidth=1.5, markersize=5)
            ax_gap.plot(RADII, gap, color=ACT_COLORS[act],
                        label=ACT_LABELS[act], marker='o',
                        linewidth=1.5, markersize=5)

        ax_cra.set_title(layer, fontsize=12)
        ax_cra.grid(alpha=0.3)
        ax_gap.grid(alpha=0.3)
        ax_gap.set_xticks(RADII)
        ax_gap.set_xticklabels(RADII_LABELS)
        ax_gap.set_xlabel('Perturbation radius')

    axes[0, 0].set_ylabel('CRA (%)')
    axes[1, 0].set_ylabel('ERA $-$ CRA (%)')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               ncol=len(ACTIVATIONS), frameon=False,
               bbox_to_anchor=(0.5, 1.0))

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Wrote {png_path}')
    print(f'  Wrote {pdf_path}')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = load_all()
    if not data:
        print('No data parsed. Aborting.', file=sys.stderr)
        sys.exit(1)
    print(f'Loaded {len(data)}/{len(LAYERS) * len(ACTIVATIONS)} cells.')
    png = os.path.join(OUT_DIR, 'cifar10_ce_cra_gap_grid.png')
    pdf = os.path.join(OUT_DIR, 'cifar10_ce_cra_gap_grid.pdf')
    plot(data, png, pdf)


if __name__ == '__main__':
    main()
