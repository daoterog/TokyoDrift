"""Benchmark definitions and invariant potentials for DW4 and LJ13."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

EnergyFunction = Callable[[torch.Tensor], torch.Tensor]


def _pair_distances(positions: torch.Tensor) -> torch.Tensor:
    """Return unique pair distances for a batch of point clouds."""
    if positions.ndim != 3:
        raise ValueError("positions must have shape [batch, particles, dimensions]")
    particles = positions.shape[1]
    rows, columns = torch.triu_indices(particles, particles, offset=1, device=positions.device)
    return (positions[:, rows] - positions[:, columns]).norm(dim=-1).clamp_min(1e-6)


def dw4_energy(positions: torch.Tensor) -> torch.Tensor:
    """Dimensionless four-particle double-well energy from Klein et al. (2023)."""
    distance = _pair_distances(positions)
    displacement = distance - 4.0
    # Equation (30): a=0, b=-4, c=0.9, d0=4, tau=1. The 1/2
    # cancels the double counting in sum_{i,j}, so unique pairs are sufficient.
    return (-4.0 * displacement.square() + 0.9 * displacement.pow(4)).sum(dim=-1)


def lj_energy(positions: torch.Tensor) -> torch.Tensor:
    """Dimensionless Lennard-Jones cluster energy with rm=epsilon=tau=1."""
    inverse_distance = _pair_distances(positions).reciprocal()
    inverse_six = inverse_distance.pow(6)
    return (inverse_six.square() - 2.0 * inverse_six).sum(dim=-1)


@dataclass(frozen=True)
class ParticleSystem:
    """Immutable definition of one fixed-size particle benchmark."""

    name: str
    particles: int
    dimensions: int
    energy: EnergyFunction
    osf_url: str
    sha256: str
    source_shape: tuple[int, int]
    paper_reference: dict[str, dict[str, float]]


SYSTEMS = {
    "dw4": ParticleSystem(
        name="dw4",
        particles=4,
        dimensions=2,
        energy=dw4_energy,
        osf_url="https://osf.io/download/mus7z/",
        sha256="60065e6c08c40e3e2d11b9fb15cbe298e457a6954af2d932d984d8ee19181b9d",
        source_shape=(1_000_000, 8),
        paper_reference={
            "likelihood": {"nll": 1.72, "ess_percent": 86.87, "path_length": 3.11},
            "ot_flow_matching": {"nll": 1.70, "ess_percent": 92.37, "path_length": 2.94},
            "equivariant_ot_flow_matching": {
                "nll": 1.68,
                "ess_percent": 88.71,
                "path_length": 2.92,
            },
        },
    ),
    "lj13": ParticleSystem(
        name="lj13",
        particles=13,
        dimensions=3,
        energy=lj_energy,
        osf_url="https://osf.io/download/bd9fg/",
        sha256="1566627762bf925a70e25d1b36da19d0fac3e2b5b86599f06e0c1c013bde6db7",
        source_shape=(10_000_000, 39),
        paper_reference={
            "likelihood": {"nll": -15.83, "ess_percent": 39.78, "path_length": 5.08},
            "ot_flow_matching": {"nll": -16.09, "ess_percent": 54.36, "path_length": 2.84},
            "equivariant_ot_flow_matching": {
                "nll": -16.07,
                "ess_percent": 57.98,
                "path_length": 2.15,
            },
        },
    ),
}


def get_system(name: str) -> ParticleSystem:
    """Return a benchmark definition by case-insensitive name."""
    try:
        return SYSTEMS[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unknown system {name!r}; choose from {sorted(SYSTEMS)}") from error


def center(positions: torch.Tensor) -> torch.Tensor:
    """Remove the geometric center from every configuration."""
    return positions - positions.mean(dim=-2, keepdim=True)


def pair_distances(positions: torch.Tensor) -> torch.Tensor:
    """Public pair-distance helper used by evaluation."""
    return _pair_distances(positions)
