"""Create the architecture and drift-normalization training matrix."""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Iterator
from pathlib import Path

VARIANTS = (
    ("egnn_unnorm", "egnn", False),
    ("egnn_norm", "egnn", True),
    ("gnn_unnorm", "gnn", False),
    ("gnn_norm", "gnn", True),
)

SYSTEM_DIMENSIONS = {
    "aldp": 3,
    "dw4": 2,
    "lj13": 3,
    "lj55": 3,
}


def trainable_parameter_count(config: dict) -> int:
    """Return the analytical trainable-parameter count for a configured model."""
    model = config["model"]
    architecture = str(model["architecture"])
    hidden_dim = int(model["hidden_dim"])
    feature_dim = int(model["feature_dim"])
    layers = int(model["layers"])
    radial_basis = int(model["radial_basis"])
    if hidden_dim <= 0 or feature_dim <= 0 or layers <= 0 or radial_basis <= 0:
        raise ValueError("model dimensions, layers, and radial_basis must be positive")

    embedding = hidden_dim**2 + feature_dim * hidden_dim + 2 * hidden_dim
    if architecture == "egnn":
        variant = str(model.get("variant", "legacy"))
        if variant not in {"legacy", "bounded"}:
            raise ValueError("model.variant must be 'legacy' or 'bounded'")
        distance_features = 2 if variant == "bounded" else 0
        per_layer = 10 * hidden_dim**2 + (radial_basis + 10 + distance_features) * hidden_dim + 2
    elif architecture == "gnn":
        try:
            dimensions = SYSTEM_DIMENSIONS[str(config["system"]).lower()]
        except KeyError as error:
            raise ValueError(
                f"cannot count GNN parameters for unknown system {config.get('system')!r}"
            ) from error
        per_layer = (
            10 * hidden_dim**2 + (radial_basis + 3 * dimensions + 9) * hidden_dim + dimensions + 1
        )
    else:
        raise ValueError("model.architecture must be 'egnn' or 'gnn'")
    return embedding + layers * per_layer


def _matched_gnn_model(config: dict) -> dict:
    """Return a GNN model with its RBF width adjusted to the EGNN count."""
    reference = copy.deepcopy(config)
    reference.setdefault("model", {})["architecture"] = "egnn"
    model = copy.deepcopy(reference["model"])
    model["architecture"] = "gnn"
    try:
        dimensions = SYSTEM_DIMENSIONS[str(config["system"]).lower()]
    except KeyError as error:
        raise ValueError(
            f"cannot match GNN parameters for unknown system {config.get('system')!r}"
        ) from error
    distance_features = 2 if str(model.get("variant", "legacy")) == "bounded" else 0
    radial_reduction = 3 * dimensions - 1 - distance_features
    model["radial_basis"] = int(model["radial_basis"]) - radial_reduction
    if model["radial_basis"] <= 0:
        raise ValueError(
            "EGNN radial_basis is too small to create a parameter-matched GNN: "
            f"it must exceed {radial_reduction}"
        )
    return model


def training_variants(config: dict) -> Iterator[tuple[str, dict]]:
    """Yield variants whose GNN capacity is matched to the input EGNN."""
    matched_gnn_model = _matched_gnn_model(config)
    for name, architecture, normalized in VARIANTS:
        variant = copy.deepcopy(config)
        if architecture == "gnn":
            variant["model"] = copy.deepcopy(matched_gnn_model)
        else:
            variant.setdefault("model", {})["architecture"] = architecture
        variant.setdefault("drift", {})["normalized"] = normalized
        yield name, variant


def write_training_variants(config_path: Path, output_directory: Path) -> list[Path]:
    """Write the four derived configs and return their paths."""
    with config_path.open() as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("training config must be a JSON object")
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, variant in training_variants(config):
        path = output_directory / f"{name}.json"
        path.write_text(json.dumps(variant, indent=2) + "\n")
        paths.append(path)
    return paths


def arguments() -> argparse.Namespace:
    """Parse config-matrix generation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Write all training variants."""
    args = arguments()
    for path in write_training_variants(args.config, args.output_directory):
        with path.open() as stream:
            variant = json.load(stream)
        print(
            f"snapshot={path} "
            f"trainable_parameters={trainable_parameter_count(variant)} "
            f"hidden_dim={variant['model']['hidden_dim']} "
            f"layers={variant['model']['layers']} "
            f"radial_basis={variant['model']['radial_basis']}"
        )


if __name__ == "__main__":
    main()
