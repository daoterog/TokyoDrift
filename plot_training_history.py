"""Plot periodic train/test metrics recorded during particle training."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from itertools import pairwise
from pathlib import Path


def arguments() -> argparse.Namespace:
    """Parse history plotting arguments."""
    parser = argparse.ArgumentParser(description="Plot train/test metrics over epochs.")
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_history(path: Path) -> list[dict]:
    """Load and validate newline-delimited train/test metric records."""
    records = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                int(record["epoch"])
                for split in ("train", "test"):
                    float(record[split]["drift_loss"])
                    float(record[split]["energy_wasserstein_1"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid history record on line {line_number}") from error
            records.append(record)
    if not records:
        raise ValueError(f"training history is empty: {path}")
    if any(
        int(current["epoch"]) <= int(previous["epoch"]) for previous, current in pairwise(records)
    ):
        raise ValueError("training history epochs must be strictly increasing")
    return records


def plot_history(records: list[dict], output: Path) -> None:
    """Write loss and energy-Wasserstein line charts."""
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "particle-drift-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    epochs = [int(record["epoch"]) for record in records]
    systems = {str(record.get("system", "particle")).upper() for record in records}
    if len(systems) != 1:
        raise ValueError("training history must contain exactly one system")
    system = systems.pop()

    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for split, label in (("train", "Train"), ("test", "Test")):
        values = [float(record[split]["drift_loss"]) for record in records]
        axis.plot(epochs, values, marker="o", markersize=3, label=label)
    axis.set(xlabel="Epoch", ylabel="Drift loss", title=f"{system} loss over training")
    if all(
        float(record[split]["drift_loss"]) > 0 for record in records for split in ("train", "test")
    ):
        axis.set_yscale("log")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output / "loss_over_epochs.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for split, label in (("train", "Train"), ("test", "Test")):
        values = [float(record[split]["energy_wasserstein_1"]) for record in records]
        axis.plot(epochs, values, marker="o", markersize=3, label=label)
    axis.set(
        xlabel="Epoch",
        ylabel="Energy Wasserstein-1 distance",
        title=f"{system} energy distribution distance over training",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output / "energy_wasserstein_over_epochs.png", dpi=180)
    plt.close(figure)


def main() -> None:
    """Load a metric history and render its charts."""
    args = arguments()
    plot_history(load_history(args.history), args.output)


if __name__ == "__main__":
    main()
