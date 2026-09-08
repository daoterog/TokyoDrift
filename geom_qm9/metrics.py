"""Alignment-free conformer ensemble metrics and geometric validity checks."""

from __future__ import annotations

import torch
from rdkit import Chem
from rdkit.Chem import rdMolAlign

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


def _molecule_with_conformers(
    atomic_numbers: torch.Tensor,
    bond_index: torch.Tensor,
    bond_order: torch.Tensor,
    conformers: torch.Tensor,
) -> Chem.Mol:
    """Create an RDKit molecule retaining the dataset's coordinate atom order."""
    editable = Chem.RWMol()
    for atomic_number in atomic_numbers.tolist():
        editable.AddAtom(Chem.Atom(int(atomic_number)))
    bond_type = {1.0: Chem.BondType.SINGLE, 1.5: Chem.BondType.AROMATIC, 2.0: Chem.BondType.DOUBLE, 3.0: Chem.BondType.TRIPLE}
    for (left, right), order in zip(bond_index.T.tolist(), bond_order.tolist(), strict=True):
        editable.AddBond(int(left), int(right), bond_type[round(float(order), 1)])
    molecule = editable.GetMol()
    for positions in conformers:
        conformer = Chem.Conformer(len(atomic_numbers))
        for index, (x, y, z) in enumerate(positions.tolist()):
            conformer.SetAtomPosition(index, (float(x), float(y), float(z)))
        molecule.AddConformer(conformer, assignId=True)
    return molecule


def symmetry_aware_cov_mat(
    generated: torch.Tensor,
    reference: torch.Tensor,
    atomic_numbers: torch.Tensor,
    bond_index: torch.Tensor,
    bond_order: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute standard COV/MAT via RDKit best RMSD over graph automorphisms.

    Rigid alignment and chemically valid atom symmetries are used only here,
    in evaluation. The model and drift objective remain alignment-free.
    """
    all_conformers = torch.cat((generated, reference))
    molecule = _molecule_with_conformers(atomic_numbers, bond_index, bond_order, all_conformers)
    generated_count = len(generated)
    distances = torch.empty(generated_count, len(reference))
    for generated_index in range(generated_count):
        for reference_index in range(len(reference)):
            distances[generated_index, reference_index] = rdMolAlign.GetBestRMS(
                molecule, molecule, generated_index, generated_count + reference_index
            )
    generated_nearest = distances.min(dim=1).values
    reference_nearest = distances.min(dim=0).values
    return {
        "cov_recall": float((reference_nearest <= threshold).float().mean()),
        "mat_recall": float(reference_nearest.mean()),
        "cov_precision": float((generated_nearest <= threshold).float().mean()),
        "mat_precision": float(generated_nearest.mean()),
    }
