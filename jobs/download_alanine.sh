#!/usr/bin/env bash
#SBATCH --job-name=alanine-data
#SBATCH --output=alanine-data-%j.out
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPOSITORY_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
if [[ ! -f "$REPOSITORY_ROOT/pyproject.toml" ]]; then
    echo "submit this job from the repository root" >&2
    exit 1
fi
UV_BIN_DIR="${UV_BIN_DIR:-$REPOSITORY_ROOT/.uv-bin}"
if command -v uv >/dev/null 2>&1; then
    UV="$(command -v uv)"
elif [[ -x "$UV_BIN_DIR/uv" ]]; then
    UV="$UV_BIN_DIR/uv"
else
    echo "uv is unavailable; run jobs/download_env.sh first" >&2
    exit 1
fi

cd "$REPOSITORY_ROOT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
# Prepare this shared environment before starting training jobs.
"$UV" sync --locked --no-group cpu --group cuda --group alanine
"$UV" run --no-sync python -m data.alanine_dipeptide.prepare "$@"
