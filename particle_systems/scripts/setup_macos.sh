#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPOSITORY_ROOT"

uv sync --project particle_systems
uv run --project particle_systems --no-sync python -m particle_systems.verify_runtime --device auto
