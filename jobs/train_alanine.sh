#!/usr/bin/env bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=train-aldp
#SBATCH --output=train-aldp-%j.out
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

RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)}}"
CONFIG="$REPOSITORY_ROOT/data/alanine_dipeptide/gaussian.json"
TRAIN_DIRECTORY="$REPOSITORY_ROOT/artifacts/checkpoints/aldp/runs/$RUN_ID"
FINAL_CHECKPOINT="$REPOSITORY_ROOT/artifacts/checkpoints/aldp/gaussian-$RUN_ID.pt"
PARAMETERS="$TRAIN_DIRECTORY/parameters.json"
RESULT_DIRECTORY="$REPOSITORY_ROOT/results/aldp/$RUN_ID"
if [[ -e "$TRAIN_DIRECTORY" || -e "$FINAL_CHECKPOINT" || -e "$RESULT_DIRECTORY" ]]; then
    echo "run $RUN_ID already exists; choose a different RUN_ID" >&2
    exit 1
fi

cd "$REPOSITORY_ROOT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-9}"
"$UV" run --no-sync \
    python -c 'import h5py, openmm; from pathlib import Path; assert Path("data/alanine_dipeptide/dataset.npz").is_file(), "Run jobs/download_alanine.sh first"'
mkdir -p "$TRAIN_DIRECTORY" "$RESULT_DIRECTORY"
exec > >(tee "$RESULT_DIRECTORY/train_and_evaluate.log") 2>&1
echo "run_id=$RUN_ID config=$CONFIG parameters=$PARAMETERS"
"$UV" run --no-sync python -m utils.verify_runtime --device cuda
"$UV" run --no-sync python -m train --config "$CONFIG" "$@" \
    --device cuda --output "$TRAIN_DIRECTORY"

if [[ ! -s "$PARAMETERS" ]]; then
    echo "training did not write $PARAMETERS" >&2
    exit 1
fi
# Only the official validation split selects the checkpoint; test is used below.
SELECTED_CHECKPOINT="$TRAIN_DIRECTORY/best_validation.pt"
if [[ ! -s "$SELECTED_CHECKPOINT" ]]; then
    SELECTED_CHECKPOINT="$TRAIN_DIRECTORY/latest.pt"
fi
echo "selected_checkpoint=$SELECTED_CHECKPOINT"
cp "$SELECTED_CHECKPOINT" "$FINAL_CHECKPOINT"
"$UV" run --no-sync python -m evaluate_alanine --checkpoint "$FINAL_CHECKPOINT" \
    --output "$RESULT_DIRECTORY" --device cuda --batch-size 256 \
    --num-samples 500000 --energy-samples 10000
echo "completed checkpoint=$FINAL_CHECKPOINT parameters=$PARAMETERS results=$RESULT_DIRECTORY"
