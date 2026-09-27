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


def training_variants(config: dict) -> Iterator[tuple[str, dict]]:
    """Yield independent configs for every architecture/normalization pair."""
    for name, architecture, normalized in VARIANTS:
        variant = copy.deepcopy(config)
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
        print(path)


if __name__ == "__main__":
    main()
