"""Fixed-topology alanine geometry in angstroms (FAB Zenodo 6993124)."""

from __future__ import annotations

import math
from typing import Any

import torch

from utils.descriptors import DescriptorDrift

ATOM_NAMES = (
    "H1",
    "CH3",
    "H2",
    "H3",
    "C",
    "O",
    "N",
    "H",
    "CA",
    "HA",
    "CB",
    "HB1",
    "HB2",
    "HB3",
    "C",
    "O",
    "N",
    "H",
    "C",
    "H1",
    "H2",
    "H3",
)
ELEMENTS = (
    "H",
    "C",
    "H",
    "H",
    "C",
    "O",
    "N",
    "H",
    "C",
    "H",
    "C",
    "H",
    "H",
    "H",
    "C",
    "O",
    "N",
    "H",
    "C",
    "H",
    "H",
    "H",
)
BONDS = (
    (0, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (4, 5),
    (4, 6),
    (6, 7),
    (6, 8),
    (8, 9),
    (8, 10),
    (8, 14),
    (10, 11),
    (10, 12),
    (10, 13),
    (14, 15),
    (14, 16),
    (16, 17),
    (16, 18),
    (18, 19),
    (18, 20),
    (18, 21),
)
PHI = (4, 6, 8, 14)
PSI = (6, 8, 14, 16)
TEMPERATURE_K = 300.0
TOPOLOGY_URL = (
    "https://raw.githubusercontent.com/choderalab/openmmtools/"
    "f6ef22a8b9f66e582df2ffa62f3bb6516de43536/"
    "openmmtools/data/alanine-dipeptide-gbsa/alanine-dipeptide.prmtop"
)
TOPOLOGY_SHA256 = "2ce81216c7e18fd4d354fac44e22ba3843d89e297884bd6389a4cd57c74ecf6e"


def labeled_distances(positions: torch.Tensor) -> torch.Tensor:
    """Keep pair identities; unlike LJ descriptors, do not sort distances."""
    if positions.ndim != 3 or positions.shape[1:] != (22, 3):
        raise ValueError("alanine positions must have shape [batch, 22, 3]")
    rows, cols = torch.triu_indices(22, 22, offset=1, device=positions.device)
    return (positions[:, rows] - positions[:, cols]).norm(dim=-1)


class AlanineDrift(DescriptorDrift):
    """Coordinate pullback of a kernel on labeled molecular pair distances."""

    @staticmethod
    def descriptors(positions: torch.Tensor) -> torch.Tensor:
        """Return ordered distances with RMS scaling and angstrom units."""
        return labeled_distances(positions) / math.sqrt(231)


def bond_lengths(positions: torch.Tensor) -> torch.Tensor:
    """Measure the 21 covalent bonds in topology order."""
    indices = torch.tensor(BONDS, device=positions.device)
    return (positions[:, indices[:, 0]] - positions[:, indices[:, 1]]).norm(dim=-1)


def angle_indices() -> list[tuple[int, int, int]]:
    """Enumerate all bond angles, counting each neighbor pair once."""
    neighbors = {i: [] for i in range(22)}
    for left, right in BONDS:
        neighbors[left].append(right)
        neighbors[right].append(left)
    return [
        (a, center, b)
        for center, atoms in neighbors.items()
        for i, a in enumerate(sorted(atoms))
        for b in sorted(atoms)[i + 1 :]
    ]


def bond_angles(positions: torch.Tensor) -> torch.Tensor:
    """Measure covalent angles in radians; degenerate angles are NaN."""
    indices = torch.tensor(angle_indices(), device=positions.device)
    a = positions[:, indices[:, 0]] - positions[:, indices[:, 1]]
    b = positions[:, indices[:, 2]] - positions[:, indices[:, 1]]
    denominator = a.norm(dim=-1) * b.norm(dim=-1)
    cosine = (a * b).sum(dim=-1) / denominator.clamp_min(1e-12)
    return cosine.clamp(-1, 1).acos().masked_fill(denominator < 1e-12, torch.nan)


def dihedral(positions: torch.Tensor, indices: tuple[int, int, int, int]) -> torch.Tensor:
    """Compute a signed periodic torsion; collinear/coincident frames are NaN."""
    a, b, c, d = (positions[:, index] for index in indices)
    axis = c - b
    axis_norm = axis.norm(dim=-1, keepdim=True)
    axis = axis / axis_norm.clamp_min(1e-12)
    left, right = a - b, d - c
    left = left - (left * axis).sum(dim=-1, keepdim=True) * axis
    right = right - (right * axis).sum(dim=-1, keepdim=True) * axis
    valid = (axis_norm[:, 0] > 1e-12) & (left.norm(dim=-1) > 1e-12) & (right.norm(dim=-1) > 1e-12)
    angle = torch.atan2(
        (torch.linalg.cross(axis, left) * right).sum(dim=-1), (left * right).sum(dim=-1)
    )
    return angle.masked_fill(~valid, torch.nan)


def backbone_angles(positions: torch.Tensor) -> torch.Tensor:
    """Return phi and psi in radians in [-pi, pi]."""
    return torch.stack((dihedral(positions, PHI), dihedral(positions, PSI)), dim=-1)


def chirality(positions: torch.Tensor) -> torch.Tensor:
    """Signed normalized tetrahedral volume at CA, with HA as the origin."""
    vectors = positions[:, [6, 14, 10]] - positions[:, 9:10]
    vectors = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.linalg.det(vectors)


def geometry_rules(training: torch.Tensor) -> dict[str, Any]:
    """Calibrate broad geometry limits exclusively on training configurations."""
    lengths, angles = bond_lengths(training), bond_angles(training)
    volumes = chirality(training)
    if not torch.isfinite(training).all() or not torch.isfinite(angles).all():
        raise ValueError("training molecular geometry contains nonfinite or degenerate values")
    sign = float(volumes.median().sign())
    if sign == 0 or float((volumes * sign > 1e-3).float().mean()) < 0.99:
        raise ValueError("training data must have a consistent, nondegenerate chirality")
    quantiles = training.new_tensor([0.001, 0.999])
    bounds = torch.quantile(lengths, quantiles, dim=0)
    angle_bounds = torch.quantile(angles, quantiles, dim=0)
    return {
        "calibration_split": "train",
        "quantiles": [0.001, 0.999],
        "bond_margin_angstrom": 0.05,
        "angle_margin_degrees": 5.0,
        "bond_lower": (bounds[0] - 0.05).clamp_min(0).tolist(),
        "bond_upper": (bounds[1] + 0.05).tolist(),
        "angle_lower": (angle_bounds[0] - math.radians(5)).clamp_min(0).tolist(),
        "angle_upper": (angle_bounds[1] + math.radians(5)).clamp_max(math.pi).tolist(),
        "chirality_sign": sign,
        "minimum_chirality_volume": 1e-3,
        "minimum_nonbonded_distance_angstrom": 0.7,
    }


def geometry_masks(positions: torch.Tensor, rules: dict) -> dict[str, torch.Tensor]:
    """Classify all frames; preserve failures in the reported denominators."""
    lengths, angles = bond_lengths(positions), bond_angles(positions)
    finite = torch.isfinite(positions).all(dim=(1, 2))
    bonds_ok = (
        (lengths >= lengths.new_tensor(rules["bond_lower"]))
        & (lengths <= lengths.new_tensor(rules["bond_upper"]))
    ).all(dim=1)
    angles_ok = (
        (angles >= angles.new_tensor(rules["angle_lower"]))
        & (angles <= angles.new_tensor(rules["angle_upper"]))
    ).all(dim=1)
    rows, cols = torch.triu_indices(22, 22, offset=1)
    nonbonded = torch.tensor(
        [(int(a), int(b)) not in BONDS for a, b in zip(rows, cols, strict=True)]
    )
    distances = labeled_distances(positions)[:, nonbonded.to(positions.device)]
    collision_free = distances.min(dim=1).values >= rules["minimum_nonbonded_distance_angstrom"]
    correct_chirality = (
        chirality(positions) * rules["chirality_sign"] > rules["minimum_chirality_volume"]
    )
    torsions_finite = torch.isfinite(backbone_angles(positions)).all(dim=1)
    geometry = finite & bonds_ok & angles_ok & collision_free & torsions_finite
    return {
        "finite": finite,
        "bonds_in_range": finite & bonds_ok,
        "angles_in_range": finite & angles_ok,
        "collision_free": finite & collision_free,
        "correct_chirality": finite & correct_chirality,
        "torsions_finite": torsions_finite,
        "geometry_valid": geometry,
        "valid": geometry & correct_chirality,
    }
