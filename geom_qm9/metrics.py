"""Alignment-free conformer ensemble metrics and geometric validity checks."""

from __future__ import annotations

import torch

from .drift import descriptor


def ensemble_metrics(generated: torch.Tensor, reference: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    """Compute COV/MAT using RMS distance-matrix error.

    This is rotation/translation invariant and avoids any training-time alignment.
    It is an explicitly labelled proxy for the usual symmetry-aware RMSD COV/MAT.
    """
    distances = torch.cdist(descriptor(generated), descriptor(reference))
    distances = distances / descriptor(generated).shape[-1] ** 0.5
    generated_nearest = distances.min(1).values
    reference_nearest = distances.min(0).values
    return {
        "coverage": float((reference_nearest <= threshold).float().mean()),
        "matching": float(generated_nearest.mean()),
        "reference_matching": float(reference_nearest.mean()),
    }


def geometry_validity(positions: torch.Tensor, bond_index: torch.Tensor, minimum_nonbonded_distance: float = 0.8) -> dict[str, float]:
    """Report no-clash fraction and bond-length summary for generated structures."""
    atoms = positions.shape[1]
    pair_distance = torch.cdist(positions, positions)
    mask = torch.triu(torch.ones(atoms, atoms, device=positions.device, dtype=torch.bool), diagonal=1)
    bonded = torch.zeros_like(mask)
    if bond_index.numel():
        bonded[bond_index[0], bond_index[1]] = True
        bonded[bond_index[1], bond_index[0]] = True
    nonbonded = pair_distance[:, mask & ~bonded]
    result = {"no_clash_fraction": float((nonbonded.min(1).values >= minimum_nonbonded_distance).float().mean())}
    if bond_index.numel():
        bonds = pair_distance[:, bond_index[0], bond_index[1]]
        result["bond_length_mean"] = float(bonds.mean())
        result["bond_length_std"] = float(bonds.std())
    return result
