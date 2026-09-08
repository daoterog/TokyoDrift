"""Visual audit for a completed GMM-40 run."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

try:  # Supports both ``python -m`` and a copied standalone toys folder.
    from .gmm40 import GMM40, Generator, metrics
except ImportError:
    from gmm40 import GMM40, Generator, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--distribution", choices=("uniform", "imbalanced"), required=True)
    parser.add_argument("--samples", type=int, default=20_000)
    args = parser.parse_args()

    checkpoint = torch.load(args.run / "checkpoint-010000.pt", map_location="cpu", weights_only=True)
    model = Generator()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    target = GMM40(torch.device("cpu"), distribution=args.distribution)
    torch.manual_seed(123)
    reference = target.sample(args.samples)
    with torch.no_grad():
        generated = model(5.0 * torch.randn(args.samples, 2))
    report = metrics(target, generated, reference)
    target_report = metrics(target, reference, reference)

    colors = ("#17324D", "#277DA1", "#E76F51")
    labels = ("frequent", "medium", "rare")
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.15), gridspec_kw={"width_ratios": (1, 1, 0.8)})
    for axis, samples, title in zip(axes[:2], (reference, generated), ("target", "one-shot drift")):
        assignment = target.mode_index(samples)
        for tier, color, label in zip(range(3), colors, labels):
            axis.scatter(
                samples[target.tiers[assignment] == tier][:, 0],
                samples[target.tiers[assignment] == tier][:, 1],
                s=0.5,
                alpha=0.18,
                color=color,
                label=label if title == "target" else None,
                rasterized=True,
            )
        axis.scatter(target.centers[:, 0], target.centers[:, 1], marker="+", s=22, linewidth=.7, color="#111111")
        axis.set(title=title, aspect="equal", xlim=(-6.2, 6.2), ylim=(-4.2, 4.2), xlabel="$x_1$")
    axes[0].set_ylabel("$x_2$")
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")

    generated_mass = torch.tensor([report[f"{name}_mass"] for name in labels])
    target_mass = torch.tensor([report[f"target_{name}_mass"] for name in labels])
    positions = torch.arange(3)
    axes[2].bar(positions - .18, target_mass, .36, label="target", color="#17324D")
    axes[2].bar(positions + .18, generated_mass, .36, label="generated", color="#E76F51")
    axes[2].set(xticks=positions, xticklabels=labels, ylim=(0, .72), ylabel="total mass", title="tier-mass audit")
    axes[2].tick_params(axis="x", labelrotation=30)
    axes[2].legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Validity: "
        f"target {100 * target_report['validity_rate']:.1f}% / generated {100 * report['validity_rate']:.1f}%"
        f" | rare-mode coverage {100 * report['rare_valid_mode_coverage']:.0f}%",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(args.run / "validity_audit.png", dpi=250, bbox_inches="tight")


if __name__ == "__main__":
    main()
