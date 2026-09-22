"""Build result-folder names from run identifiers and training configs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")


def result_folder_name(config: dict, run_id: str) -> str:
    """Return ``id_architecture_(un)norm_(no)descr`` for a training run."""
    try:
        architecture = config["model"]["architecture"]
        normalized = config["drift"]["normalized"]
        descriptors = config["drift"]["descriptors"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "config must define model.architecture, drift.normalized, and drift.descriptors"
        ) from error

    if not isinstance(architecture, str) or not SAFE_COMPONENT.fullmatch(architecture):
        raise ValueError("model.architecture must be a safe folder-name component")
    if type(normalized) is not bool:
        raise ValueError("drift.normalized must be true or false")
    if type(descriptors) is not bool:
        raise ValueError("drift.descriptors must be true or false")
    if not run_id or not SAFE_COMPONENT.fullmatch(run_id):
        raise ValueError("run id must contain only letters, numbers, dots, underscores, and hyphens")

    normalization = "norm" if normalized else "unnorm"
    descriptor_mode = "descr" if descriptors else "nodescr"
    return f"{run_id}_{architecture}_{normalization}_{descriptor_mode}"


def arguments() -> argparse.Namespace:
    """Parse the config and run identifier used by the job scripts."""
    parser = argparse.ArgumentParser(description="Construct a result-folder name.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--id", required=True, dest="run_id")
    return parser.parse_args()


def main() -> None:
    """Print the validated result-folder name."""
    args = arguments()
    with args.config.open() as stream:
        config = json.load(stream)
    print(result_folder_name(config, args.run_id))


if __name__ == "__main__":
    main()
