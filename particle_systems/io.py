"""Configuration, dataset, model, and device loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import ParticleGenerator
from .systems import center, get_system


def _mps_available() -> bool:
    """Return whether this PyTorch build exposes an available MPS backend."""
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def load_config(path: Path) -> dict[str, Any]:
    """Load a JSON experiment configuration and validate its system name."""
    with path.open() as stream:
        config = json.load(stream)
    get_system(config["system"])
    return config


def load_dataset(path: Path, split: str) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load a centered train or test split from a prepared archive."""
    with np.load(path, allow_pickle=False) as archive:
        values = torch.from_numpy(np.asarray(archive[split], dtype=np.float32))
        metadata = json.loads(str(archive["metadata"].item()))
    system = get_system(metadata["system"])
    expected = (system.particles, system.dimensions)
    if tuple(values.shape[1:]) != expected:
        raise ValueError(
            f"expected {split} samples ending in {expected}, got {tuple(values.shape)}"
        )
    return center(values), metadata


def build_model(config: dict[str, Any]) -> ParticleGenerator:
    """Construct a particle generator from a resolved configuration."""
    model = config["model"]
    if config["system"] == "aldp" and model.get("fixed_atom_identity") is not True:
        raise ValueError(
            "alanine requires model.fixed_atom_identity: true for its labeled topology"
        )
    return ParticleGenerator(
        feature_dim=int(model["feature_dim"]),
        hidden_dim=int(model["hidden_dim"]),
        layers=int(model["layers"]),
        radial_basis=int(model["radial_basis"]),
        max_distance=float(model["max_distance"]),
        fixed_atom_identity=(
            get_system(config["system"]).particles
            if model.get("fixed_atom_identity", False)
            else None
        ),
    )


def select_device(requested: str) -> torch.device:
    """Resolve an explicit device or choose CUDA, MPS, then CPU."""
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is false. "
                "Install the CUDA dependency group and verify the NVIDIA driver."
            )
        if device.type == "mps" and not _mps_available():
            raise RuntimeError("MPS was requested but is not available in this PyTorch runtime.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_summary(device: torch.device) -> dict[str, str | bool | int | None]:
    """Return serializable runtime details for logs and reproducibility."""
    summary: dict[str, str | bool | int | None] = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_name": None,
    }
    if device.type == "cuda":
        summary["device_name"] = torch.cuda.get_device_name(device)
        summary["cuda_device_count"] = torch.cuda.device_count()
    elif device.type == "mps":
        summary["device_name"] = "Apple Metal Performance Shaders"
    else:
        summary["device_name"] = "CPU"
    return summary
