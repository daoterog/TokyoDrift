#!/usr/bin/env bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=train-lj13
#SBATCH --output=train-lj13-%j.out
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=32G

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
if command -v uv >/dev/null 2>&1; then
    UV="$(command -v uv)"
elif [[ -x "$UV_BIN_DIR/uv" ]]; then
    UV="$UV_BIN_DIR/uv"
else
    echo "uv is unavailable; run jobs/download_env.sh first" >&2
    exit 1
fi

RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)}}"
CONFIG="$REPOSITORY_ROOT/data/lj13/gaussian.json"
RUN_DIRECTORY="$REPOSITORY_ROOT/results/lj13/$RUN_ID"
CHECKPOINT_DIRECTORY="$RUN_DIRECTORY/checkpoints"
FINAL_CHECKPOINT="$CHECKPOINT_DIRECTORY/final.pt"
PARAMETERS="$CHECKPOINT_DIRECTORY/parameters.json"

if [[ -e "$RUN_DIRECTORY" ]]; then
    echo "run $RUN_ID already exists; choose a different RUN_ID" >&2
    exit 1
fi
mkdir -p "$CHECKPOINT_DIRECTORY"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    SLURM_OUTPUT="$REPOSITORY_ROOT/train-lj13-$SLURM_JOB_ID.out"
    if [[ -e "$SLURM_OUTPUT" ]]; then
        mv "$SLURM_OUTPUT" "$RUN_DIRECTORY/slurm.out"
    fi
fi
exec > >(tee "$RUN_DIRECTORY/train_and_evaluate.log") 2>&1

cd "$REPOSITORY_ROOT"
export PYTHONUNBUFFERED=1

echo "run_id=$RUN_ID"
echo "training_config=$CONFIG"
echo "run_output=$RUN_DIRECTORY"
echo "checkpoint_output=$CHECKPOINT_DIRECTORY"
echo "final_checkpoint=$FINAL_CHECKPOINT"
echo "parameters=$PARAMETERS"
echo "evaluation_output=$RUN_DIRECTORY"

"$UV" run --no-sync python -m utils.verify_runtime --device cuda

"$UV" run --no-sync python -m train \
    --config "$CONFIG" \
    --device cuda \
    --output "$CHECKPOINT_DIRECTORY"

if [[ ! -s "$PARAMETERS" ]]; then
    echo "training completed without a parameter snapshot at $PARAMETERS" >&2
    exit 1
fi
cp "$CHECKPOINT_DIRECTORY/latest.pt" "$FINAL_CHECKPOINT"

"$UV" run --no-sync python -m evaluate \
    --checkpoint "$FINAL_CHECKPOINT" \
    --output "$RUN_DIRECTORY" \
    --device cuda

echo "completed checkpoint=$FINAL_CHECKPOINT parameters=$PARAMETERS results=$RUN_DIRECTORY"
