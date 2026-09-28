#!/usr/bin/env bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=trial-dw4-block
#SBATCH --output=train-dw4-%j.out
#SBATCH --time=1:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=32G

# Trial of the block EGNN on DW4 with the best previous training setup
# (schedule_lr_longer, normalized drift). Snapshots configs/dw4_config.json with
# drift.normalized=true, then trains and evaluates through jobs/train_dw4.sh.
#
#   sbatch jobs/trial_dw4_block.sh                  # full run
#   sbatch jobs/trial_dw4_block.sh --epochs 20      # smoke test; args go to train.py
#   RUN_ID=my_trial sbatch jobs/trial_dw4_block.sh  # custom result folder prefix

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPOSITORY_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
if [[ ! -f "$REPOSITORY_ROOT/jobs/train_dw4.sh" ]]; then
    echo "repository not found at $REPOSITORY_ROOT; submit this job from the repository root" >&2
    exit 1
fi

export RUN_ID="${RUN_ID:-block_trial_${SLURM_JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)}}"
SNAPSHOT_DIRECTORY="$REPOSITORY_ROOT/results/submitted-configs/dw4/$RUN_ID"
export CONFIG_SNAPSHOT="$SNAPSHOT_DIRECTORY/egnn_norm.json"
if [[ -e "$SNAPSHOT_DIRECTORY" ]]; then
    echo "config snapshot $SNAPSHOT_DIRECTORY already exists; choose a different RUN_ID" >&2
    exit 1
fi
mkdir -p "$SNAPSHOT_DIRECTORY"
python3 - "$REPOSITORY_ROOT/configs/dw4_config.json" "$CONFIG_SNAPSHOT" <<'EOF'
import json
import sys

source, target = sys.argv[1:]
with open(source) as stream:
    config = json.load(stream)
config["drift"]["normalized"] = True
with open(target, "w") as stream:
    json.dump(config, stream, indent=2)
    stream.write("\n")
EOF
chmod a-w "$CONFIG_SNAPSHOT"
echo "config_snapshot=$CONFIG_SNAPSHOT"

# Run in this allocation; train_dw4.sh moves train-dw4-$SLURM_JOB_ID.out into the run folder.
exec bash "$REPOSITORY_ROOT/jobs/train_dw4.sh" "$@"
