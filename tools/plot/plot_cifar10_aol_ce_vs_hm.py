"""1x2 bar chart: CIFAR-10 + AOL, CE vs HM, across activations."""

import os
import re

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = os.path.join("results", "CIFAR10")
OUTPUT_DIR = os.path.join("results", "figures")

ACTS = ["AV", "GS", "HH", "LS", "NA"]
ACT_LABELS = ["AbsValue", "MaxMin", "Householder", "LinSpline", "N-act"]
LOSSES = ["CE", "HM"]


def parse_metrics(path):
    with open(path) as f:
        text = f.read()
    clean = float(re.search(r"clean accuracy\s*=\s*([\d.]+)%", text).group(1))
    cra = float(re.search(r"36/255\s+([\d.]+)%", text).group(1))
    return clean, cra


def main():
    clean = {loss: [] for loss in LOSSES}
    cra = {loss: [] for loss in LOSSES}
    for act in ACTS:
        for loss in LOSSES:
            path = os.path.join(RESULTS_DIR, f"XS_AOL_{act}_{loss}.txt")
            c, r = parse_metrics(path)
            clean[loss].append(c)
            cra[loss].append(r)

    x = np.arange(len(ACTS))
    width = 0.4

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.4),
                                     constrained_layout=True)

    for ax, data, ylabel, ymax in [
        (ax_l, clean, "Clean accuracy (%)", 80),
        (ax_r, cra, "CRA @ 36/255 (%)", 60),
    ]:
        b_ce = ax.bar(x - width / 2, data["CE"], width, label="CE",
                      color="#3d85bd")
        b_hm = ax.bar(x + width / 2, data["HM"], width, label="HM",
                      color="#ee9138")
        ax.bar_label(b_ce, fmt="%.1f", padding=2, fontsize=8)
        ax.bar_label(b_hm, fmt="%.1f", padding=2, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(ACT_LABELS)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf_path = os.path.join(OUTPUT_DIR, "cifar10_aol_ce_vs_hm.pdf")
    png_path = os.path.join(OUTPUT_DIR, "cifar10_aol_ce_vs_hm.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
