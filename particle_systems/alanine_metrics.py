"""Periodic backbone and topology-aware geometry metrics for alanine sampling."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from .alanine import backbone_angles, bond_angles, bond_lengths, geometry_masks


def probability_metrics(reference: np.ndarray, generated: np.ndarray) -> dict:
    """Report JS, forward KL and total variation for fixed-bin probability masses."""
    if reference.sum() == 0 or generated.sum() == 0:
        return {"js": None, "kl_reference_to_generated": None, "total_variation": None}
    p, q = reference.flatten().astype(float), generated.flatten().astype(float)
    p, q = p / p.sum(), q / q.sum()
    total_variation = float(np.abs(p - q).sum() / 2)
    # Same probability floor for both populations; report this in the definitions.
    p, q = np.maximum(p, 1e-10), np.maximum(q, 1e-10)
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    return {
        "js": float((np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))) / 2),
        "kl_reference_to_generated": float(np.sum(p * np.log(p / q))),
        "total_variation": total_variation,
    }


def torsion_histogram(angles: torch.Tensor, bins: int = 64) -> np.ndarray:
    """Histogram the torus with shared fixed edges; +pi and -pi are identical."""
    angles = angles[torch.isfinite(angles).all(dim=1)].detach().cpu().numpy()
    wrapped = (angles + np.pi) % (2 * np.pi) - np.pi
    return np.histogram2d(
        wrapped[:, 0], wrapped[:, 1], bins=bins, range=[[-np.pi, np.pi], [-np.pi, np.pi]]
    )[0]


def torsion_comparison(generated: torch.Tensor, reference: torch.Tensor) -> dict:
    """Compare raw or explicitly conditioned phi/psi distributions."""
    gen, ref = backbone_angles(generated), backbone_angles(reference)
    gen_hist, ref_hist = torsion_histogram(gen), torsion_histogram(ref)
    return {
        "generated_count": len(generated),
        "reference_count": len(reference),
        "generated_finite_torsion_count": int(torch.isfinite(gen).all(dim=1).sum()),
        "reference_finite_torsion_count": int(torch.isfinite(ref).all(dim=1).sum()),
        "phi_psi": probability_metrics(ref_hist, gen_hist),
        "phi": probability_metrics(ref_hist.sum(axis=1), gen_hist.sum(axis=1)),
        "psi": probability_metrics(ref_hist.sum(axis=0), gen_hist.sum(axis=0)),
    }


def exact_wasserstein_1(reference: np.ndarray, generated: np.ndarray) -> float | None:
    """Integrate empirical CDF differences exactly, including extreme tails."""
    p, q = np.sort(np.asarray(reference).flatten()), np.sort(np.asarray(generated).flatten())
    p, q = p[np.isfinite(p)], q[np.isfinite(q)]
    if not len(p) or not len(q):
        return None
    grid = np.sort(np.concatenate((p, q)))
    cdf_p = np.searchsorted(p, grid[:-1], side="right") / len(p)
    cdf_q = np.searchsorted(q, grid[:-1], side="right") / len(q)
    return float(np.sum(np.abs(cdf_p - cdf_q) * np.diff(grid)))


def subset_indices(count: int, maximum: int, seed: int = 17903) -> np.ndarray:
    """Choose whole configurations deterministically without replacement."""
    if count < 0 or maximum <= 0:
        raise ValueError("sample counts must be non-negative and the cap positive")
    if count <= maximum:
        return np.arange(count)
    return np.sort(np.random.default_rng(seed).choice(count, maximum, replace=False))


def geometry_report(
    generated: torch.Tensor, reference: torch.Tensor, rules: dict
) -> dict[str, Any]:
    """Keep validity denominators on all frames, with separate filtered torsion metrics."""
    gen_masks, ref_masks = geometry_masks(generated, rules), geometry_masks(reference, rules)
    return {
        "definition": rules,
        "generated": {
            name + "_fraction": float(mask.double().mean()) for name, mask in gen_masks.items()
        },
        "reference": {
            name + "_fraction": float(mask.double().mean()) for name, mask in ref_masks.items()
        },
        "valid_counts": {
            "generated": int(gen_masks["valid"].sum()),
            "reference": int(ref_masks["valid"].sum()),
        },
        "raw_torsions": torsion_comparison(generated, reference),
        "valid_torsions": torsion_comparison(
            generated[gen_masks["valid"]], reference[ref_masks["valid"]]
        ),
    }


def observable_report(generated: torch.Tensor, reference: torch.Tensor, cap: int = 20_000) -> dict:
    """Compare labeled bond/angle marginals on a declared deterministic subset."""
    generated = generated[torch.as_tensor(subset_indices(len(generated), cap))]
    reference = reference[torch.as_tensor(subset_indices(len(reference), cap))]
    results: dict[str, Any] = {"generated_count": len(generated), "reference_count": len(reference)}
    for name, function in (
        ("bond_length_angstrom", bond_lengths),
        ("bond_angle_radians", bond_angles),
    ):
        gen, ref = function(generated).numpy(), function(reference).numpy()
        values = [exact_wasserstein_1(ref[:, i], gen[:, i]) for i in range(gen.shape[1])]
        available = [value for value in values if value is not None]
        results[name] = {
            "per_observable_w1": values,
            "mean_w1": float(np.mean(available)) if available else None,
            "max_w1": max(available) if available else None,
        }
    return results


def validation_metrics(generated: torch.Tensor, reference: torch.Tensor, rules: dict) -> dict:
    """Select checkpoints on held-out torsions and failures, without expensive energies."""
    report = geometry_report(generated, reference, rules)
    js = report["raw_torsions"]["phi_psi"]["js"]
    valid = report["generated"]["valid_fraction"]
    return {
        "phi_psi_js": js if js is not None else math.log(2),
        "valid_fraction": valid,
        "selection_score": (js if js is not None else math.log(2)) + 1 - valid,
    }
