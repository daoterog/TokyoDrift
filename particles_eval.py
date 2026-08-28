"""Evaluation for generated DW4 / LJ13 configurations.

These systems have no FID. What the equivariant-flow literature reports instead
is distributional agreement in physically meaningful, symmetry-invariant
coordinates: the histogram of interparticle distances and the histogram of
energies, generated samples against held-out data.

That choice is not cosmetic. A configuration is defined only up to translation,
global rotation and particle relabelling, so any metric computed on raw
coordinates measures the arbitrary pose as much as the physics. Every quantity
here is computed from pairwise distances, which are invariant to all three.

`sorted_distance_features` is in this module for the same reason but is not only
an evaluation tool: it is the natural invariant feature map to drift in, since
the drifting kernel exp(-||x - y|| / tau) compares whole configurations and
would otherwise call a rotated copy of a sample dissimilar to itself.
"""

import torch

from .particles import LennardJonesPotential, MultiDoubleWellPotential, pairwise_distances

Energy = MultiDoubleWellPotential | LennardJonesPotential


def sorted_distance_features(x: torch.Tensor) -> torch.Tensor:
    """Compute a permutation-, rotation- and translation-invariant descriptor.

    Sorting the pairwise distances removes the dependence on particle ordering;
    using distances at all removes the dependence on pose.

    Args:
        x: Positions of shape ``[batch, n_particles, n_dimensions]``.

    Returns:
        Sorted distances of shape ``[batch, n_particles * (n_particles - 1) // 2]``.
    """
    return pairwise_distances(x).sort(dim=-1).values


def wasserstein1(a: torch.Tensor, b: torch.Tensor, n_quantiles: int = 1_000) -> float:
    """Compute the 1-Wasserstein distance between two 1-D empirical samples.

    Uses the quantile-function form, so the two samples need not have the same
    size. Implemented directly rather than via `torch.quantile`, which refuses
    inputs above a size limit that distance histograms exceed easily.

    Args:
        a: First sample, any shape; flattened.
        b: Second sample, any shape; flattened.
        n_quantiles: Number of quantiles used to approximate the integral.

    Returns:
        The 1-Wasserstein distance.
    """
    grid = torch.linspace(0.0, 1.0, n_quantiles, dtype=torch.float64)
    quantiles = []
    for sample in (a, b):
        sorted_sample = sample.flatten().double().sort().values
        positions = grid * (len(sorted_sample) - 1)
        low = positions.floor().long()
        high = positions.ceil().long()
        weight = positions - low
        quantiles.append(
            sorted_sample[low] * (1 - weight) + sorted_sample[high] * weight
        )
    return (quantiles[0] - quantiles[1]).abs().mean().item()


def evaluate_samples(
    samples: torch.Tensor,
    reference: torch.Tensor,
    energy: Energy,
) -> dict[str, float]:
    """Score generated configurations against held-out data.

    Args:
        samples: Generated positions, shape ``[n, n_particles, n_dimensions]``.
        reference: Held-out data positions, same trailing shape.
        energy: The system's analytic energy.

    Returns:
        Dict of metrics. `distance_w1` is the headline number (lower is better,
        0 means the histograms coincide); the rest describe what the samples
        look like and are read against the reference columns.

        Read `energy_w1` together with `clash_rate`, never alone. The LJ pair
        term grows as r^-12, so one collapsed pair in one sample moves
        `energy_w1` by many orders of magnitude and it stops ranking models.
        `clash_rate` is the interpretable version of the same failure.

        Note also that `distance_w1` has a nonzero floor on these benchmarks:
        val against test scores ~0.02 for DW4 and ~0.09 for LJ13, because the
        splits are consecutive blocks of a correlated MCMC chain rather than
        independent draws. Compare a model against that floor, not against 0.
    """
    finite = torch.isfinite(samples).all(dim=(-2, -1))
    n_invalid = int((~finite).sum())
    samples = samples[finite]

    sample_dists = pairwise_distances(samples)
    reference_dists = pairwise_distances(reference)
    sample_energies = energy(samples)
    reference_energies = energy(reference)

    return {
        "n_samples": float(len(samples)),
        "n_invalid": float(n_invalid),
        "distance_w1": wasserstein1(sample_dists, reference_dists),
        "energy_w1": wasserstein1(sample_energies, reference_energies),
        "energy_mean": sample_energies.mean().item(),
        "energy_std": sample_energies.std().item(),
        "reference_energy_mean": reference_energies.mean().item(),
        "reference_energy_std": reference_energies.std().item(),
        # A collapsed pair sends the LJ energy to +inf, so the smallest distance
        # is the early-warning signal that samples are physically broken.
        "min_distance": sample_dists.min().item(),
        "reference_min_distance": reference_dists.min().item(),
        # Fraction of samples that fall outside the energy range the data
        # occupies at all -- bounded, so it stays readable when energy_w1 blows up.
        "clash_rate": (sample_energies > reference_energies.max()).float().mean().item(),
    }


def plot_comparison(
    samples: torch.Tensor,
    reference: torch.Tensor,
    energy: Energy,
    n_bins: int = 100,
    labels: tuple[str, str] = ("generated", "data"),
):
    """Plot distance and energy histograms, generated against data.

    This is the figure the bgflow notebooks produce for these systems: overlaid
    step histograms in the invariant coordinates, which show *where* a model is
    wrong (a missing mode, an over-broad well) in a way a scalar cannot.

    Args:
        samples: Generated positions, shape ``[n, n_particles, n_dimensions]``.
        reference: Held-out data positions, same trailing shape.
        energy: The system's analytic energy.
        n_bins: Number of histogram bins.
        labels: Legend labels for (samples, reference).

    Returns:
        The matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    fig, (dist_ax, energy_ax) = plt.subplots(1, 2, figsize=(13, 4.5))

    sample_dists = pairwise_distances(samples).flatten()
    reference_dists = pairwise_distances(reference).flatten()
    dist_range = (
        min(sample_dists.min().item(), reference_dists.min().item()),
        max(sample_dists.max().item(), reference_dists.max().item()),
    )
    for values, label in [(sample_dists, labels[0]), (reference_dists, labels[1])]:
        dist_ax.hist(
            values.cpu().numpy(),
            bins=n_bins,
            range=dist_range,
            density=True,
            histtype="step",
            linewidth=2,
            label=label,
        )
    dist_ax.set_xlabel("interparticle distance")
    dist_ax.set_ylabel("density")
    dist_ax.legend()

    # Generated samples can land at absurd energies, which would flatten the
    # plot; clip the axis to the range the data actually occupies.
    reference_energies = energy(reference)
    sample_energies = energy(samples)
    lo = reference_energies.min().item()
    hi = reference_energies.max().item()
    pad = 0.5 * (hi - lo)
    energy_range = (lo - pad, hi + pad)
    for values, label in [(sample_energies, labels[0]), (reference_energies, labels[1])]:
        energy_ax.hist(
            values.cpu().numpy(),
            bins=n_bins,
            range=energy_range,
            density=True,
            histtype="step",
            linewidth=2,
            label=label,
        )
    energy_ax.set_xlabel("energy")
    energy_ax.set_ylabel("density")
    energy_ax.legend()

    fig.tight_layout()
    return fig
