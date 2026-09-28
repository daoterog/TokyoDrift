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


EGNN_ONLY_KEYS = (
    "variant",
    "coordinate_range",
    "layers_per_block",
    "attention",
    "tanh_coordinate_updates",
    "aggregation",
)


def _system_dimensions(config: dict) -> int:
    try:
        return SYSTEM_DIMENSIONS[str(config["system"]).lower()]
    except KeyError as error:
        raise ValueError(
            f"cannot count GNN parameters for unknown system {config.get('system')!r}"
        ) from error


def trainable_parameter_count(config: dict) -> int:
    """Return the analytical trainable-parameter count for a configured model."""
    model = config["model"]
    architecture = str(model["architecture"])
    hidden_dim = int(model["hidden_dim"])
    feature_dim = int(model["feature_dim"])
    layers = int(model["layers"])
    if hidden_dim <= 0 or feature_dim <= 0 or layers <= 0:
        raise ValueError("model dimensions and layers must be positive")
    variant = str(model.get("variant", "legacy")) if architecture == "egnn" else None

    if variant == "block":
        layers_per_block = int(model.get("layers_per_block", 1))
        if layers_per_block <= 0:
            raise ValueError("model.layers_per_block must be positive")
        attention = (hidden_dim + 1) if bool(model.get("attention", True)) else 0
        type_gcn = 9 * hidden_dim**2 + 15 * hidden_dim + 6 + attention
        pos_gcn = 3 * hidden_dim**2 + 5 * hidden_dim
        embedding = feature_dim * hidden_dim + hidden_dim
        return embedding + layers * (layers_per_block * type_gcn + pos_gcn)

    radial_basis = int(model["radial_basis"])
    if radial_basis <= 0:
        raise ValueError("model.radial_basis must be positive")
    embedding = hidden_dim**2 + feature_dim * hidden_dim + 2 * hidden_dim
    if architecture == "egnn":
        if variant not in {"legacy", "bounded"}:
            raise ValueError("model.variant must be 'block', 'legacy' or 'bounded'")
        distance_features = 2 if variant == "bounded" else 0
        per_layer = 10 * hidden_dim**2 + (radial_basis + 10 + distance_features) * hidden_dim + 2
    elif architecture == "gnn":
        dimensions = _system_dimensions(config)
        per_layer = (
            10 * hidden_dim**2 + (radial_basis + 3 * dimensions + 9) * hidden_dim + dimensions + 1
        )
    else:
        raise ValueError("model.architecture must be 'egnn' or 'gnn'")
    return embedding + layers * per_layer


def _matched_gnn_model(config: dict) -> dict:
    """Return a GNN model whose RBF width best matches the EGNN parameter count."""
    reference = copy.deepcopy(config)
    reference.setdefault("model", {})["architecture"] = "egnn"
    target = trainable_parameter_count(reference)
    model = {
        key: value for key, value in reference["model"].items() if key not in EGNN_ONLY_KEYS
    }
    model["architecture"] = "gnn"
    # GNN parameters grow by layers * hidden_dim per radial channel.
    model["radial_basis"] = 1
    base = trainable_parameter_count({**config, "model": model}) - int(model["layers"]) * int(
        model["hidden_dim"]
    )
    radial_basis = round((target - base) / (int(model["layers"]) * int(model["hidden_dim"])))
    if radial_basis <= 0:
        raise ValueError(
            "EGNN has too few parameters to create a parameter-matched GNN: "
            f"the matched GNN would need {radial_basis} radial channels"
        )
    model["radial_basis"] = radial_basis
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
            f"radial_basis={variant['model'].get('radial_basis', 'none')}"
        )


if __name__ == "__main__":
    main()
