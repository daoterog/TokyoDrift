#!/usr/bin/env bash
#SBATCH --job-name=particle-env
#SBATCH --output=particle-env-%j.out
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPOSITORY_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
if [[ ! -f "$REPOSITORY_ROOT/pyproject.toml" ]]; then
    echo "repository not found at $REPOSITORY_ROOT; submit this job from the repository root" >&2
    exit 1
fi
UV_BIN_DIR="${UV_BIN_DIR:-$REPOSITORY_ROOT/.uv-bin}"
DEVICE_GROUP="${1:-${PARTICLE_DEVICE_GROUP:-cuda}}"

case "$DEVICE_GROUP" in
    cuda)
        SYNC_GROUPS=(--no-group cpu --group cuda)
        ;;
    cpu)
        SYNC_GROUPS=(--no-group cuda --group cpu)
        ;;
    *)
        echo "usage: $0 [cuda|cpu]" >&2
        exit 2
        ;;
esac

if command -v uv >/dev/null 2>&1; then
    UV="$(command -v uv)"
else
    if ! command -v curl >/dev/null 2>&1; then
        echo "curl is required to install uv" >&2
        exit 1
    fi
    mkdir -p "$UV_BIN_DIR"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UV_BIN_DIR" sh
    UV="$UV_BIN_DIR/uv"
fi

cd "$REPOSITORY_ROOT"
"$UV" sync --locked "${SYNC_GROUPS[@]}"
