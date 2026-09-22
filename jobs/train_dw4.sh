#!/usr/bin/env bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=train-dw4
#SBATCH --output=train-dw4-%j.out
#SBATCH --time=1:00:00
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
CONFIG="$REPOSITORY_ROOT/configs/dw4_config.json"
cd "$REPOSITORY_ROOT"
RUN_NAME="$("$UV" run --no-sync python -m utils.result_naming --config "$CONFIG" --id "$RUN_ID")"
RUN_DIRECTORY="$REPOSITORY_ROOT/results/dw4/$RUN_NAME"
CHECKPOINT_DIRECTORY="$RUN_DIRECTORY/checkpoints"
FINAL_CHECKPOINT="$CHECKPOINT_DIRECTORY/final.pt"
PARAMETERS="$CHECKPOINT_DIRECTORY/parameters.json"

if [[ -e "$RUN_DIRECTORY" ]]; then
    echo "run $RUN_NAME already exists; choose a different RUN_ID" >&2
    exit 1
fi
mkdir -p "$CHECKPOINT_DIRECTORY"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    SLURM_OUTPUT="$REPOSITORY_ROOT/train-dw4-$SLURM_JOB_ID.out"
    if [[ -e "$SLURM_OUTPUT" ]]; then
        mv "$SLURM_OUTPUT" "$RUN_DIRECTORY/slurm.out"
    fi
fi
exec > >(tee "$RUN_DIRECTORY/train_and_evaluate.log") 2>&1

export PYTHONUNBUFFERED=1

echo "run_id=$RUN_ID"
echo "run_name=$RUN_NAME"
echo "training_config=$CONFIG"
echo "run_output=$RUN_DIRECTORY"
echo "checkpoint_output=$CHECKPOINT_DIRECTORY"
echo "final_checkpoint=$FINAL_CHECKPOINT"
echo "parameters=$PARAMETERS"
echo "evaluation_output=$RUN_DIRECTORY"

"$UV" run --no-sync python -m train \
    --config "$CONFIG" \
    --device cuda \
    --output "$CHECKPOINT_DIRECTORY"

if [[ ! -s "$PARAMETERS" ]]; then
    echo "training completed without a parameter snapshot at $PARAMETERS" >&2
    exit 1
fi
# Select only on the held-out training subset configured in dw4_config.json.
# Periodic test metrics are diagnostic and never participate in selection.
SELECTED_CHECKPOINT="$CHECKPOINT_DIRECTORY/best_validation.pt"
if [[ ! -s "$SELECTED_CHECKPOINT" ]]; then
    SELECTED_CHECKPOINT="$CHECKPOINT_DIRECTORY/latest.pt"
fi
echo "selected_checkpoint=$SELECTED_CHECKPOINT"
cp "$SELECTED_CHECKPOINT" "$FINAL_CHECKPOINT"

TRAIN_TEST_HISTORY="$CHECKPOINT_DIRECTORY/train_test_history.jsonl"
if [[ ! -s "$TRAIN_TEST_HISTORY" ]]; then
    echo "training completed without train/test history at $TRAIN_TEST_HISTORY" >&2
    exit 1
fi
"$UV" run --no-sync python -m plot_training_history \
    --history "$TRAIN_TEST_HISTORY" \
    --output "$RUN_DIRECTORY"

"$UV" run --no-sync python -m evaluate \
    --checkpoint "$FINAL_CHECKPOINT" \
    --output "$RUN_DIRECTORY" \
    --device cuda

echo "completed checkpoint=$FINAL_CHECKPOINT parameters=$PARAMETERS results=$RUN_DIRECTORY"
