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
TRAIN_DIRECTORY="$REPOSITORY_ROOT/artifacts/checkpoints/lj13/runs/$RUN_ID"
FINAL_CHECKPOINT="$REPOSITORY_ROOT/artifacts/checkpoints/lj13/gaussian-$RUN_ID.pt"
PARAMETERS="$TRAIN_DIRECTORY/parameters.json"
RESULT_DIRECTORY="$REPOSITORY_ROOT/results/lj13/$RUN_ID"

if [[ -e "$TRAIN_DIRECTORY" || -e "$FINAL_CHECKPOINT" || -e "$RESULT_DIRECTORY" ]]; then
    echo "run $RUN_ID already has model or result output; choose a different RUN_ID" >&2
    exit 1
fi
mkdir -p "$TRAIN_DIRECTORY" "$RESULT_DIRECTORY"
exec > >(tee "$RESULT_DIRECTORY/train_and_evaluate.log") 2>&1

cd "$REPOSITORY_ROOT"
export PYTHONUNBUFFERED=1

echo "run_id=$RUN_ID"
echo "training_config=$CONFIG"
echo "training_output=$TRAIN_DIRECTORY"
echo "final_checkpoint=$FINAL_CHECKPOINT"
echo "parameters=$PARAMETERS"
echo "evaluation_output=$RESULT_DIRECTORY"

"$UV" run --no-sync python -m utils.verify_runtime --device cuda

"$UV" run --no-sync python -m train \
    --config "$CONFIG" \
    --device cuda \
    --output "$TRAIN_DIRECTORY"

if [[ ! -s "$PARAMETERS" ]]; then
    echo "training completed without a parameter snapshot at $PARAMETERS" >&2
    exit 1
fi
cp "$TRAIN_DIRECTORY/latest.pt" "$FINAL_CHECKPOINT"

"$UV" run --no-sync python -m evaluate \
    --checkpoint "$FINAL_CHECKPOINT" \
    --output "$RESULT_DIRECTORY" \
    --device cuda

echo "completed checkpoint=$FINAL_CHECKPOINT parameters=$PARAMETERS results=$RESULT_DIRECTORY"
