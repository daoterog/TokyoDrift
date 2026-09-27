#!/usr/bin/env bash
# Snapshot one training config and submit the matching Slurm job.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: jobs/submit_training.sh SYSTEM [--config PATH] [--run-id ID] [-- TRAIN_ARGS...]

SYSTEM may be dw4, lj13, lj55, or alanine_dipeptide. The selected config is copied to a
run-specific, read-only snapshot before sbatch is called. Arguments after -- are forwarded to
the training command when the job starts.
EOF
}

if [[ $# -eq 0 ]]; then
    usage >&2
    exit 2
fi

SYSTEM="$1"
shift
case "$SYSTEM" in
    dw4|lj13|lj55)
        CONFIG_NAME="${SYSTEM}_config.json"
        JOB_NAME="train_${SYSTEM}.sh"
        RESULT_SYSTEM="$SYSTEM"
        ;;
    aldp|alanine|alanine_dipeptide)
        CONFIG_NAME="alanine_dipeptide_config.json"
        JOB_NAME="train_alanine.sh"
        RESULT_SYSTEM="alanine_dipeptide"
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "unknown system $SYSTEM; choose dw4, lj13, lj55, or alanine_dipeptide" >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="$REPOSITORY_ROOT/configs/$CONFIG_NAME"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%S)-$$-$RANDOM}"
TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            if [[ $# -lt 2 ]]; then
                echo "--config requires a path" >&2
                exit 2
            fi
            CONFIG="$2"
            shift 2
            ;;
        --run-id)
            if [[ $# -lt 2 ]]; then
                echo "--run-id requires a value" >&2
                exit 2
            fi
            RUN_ID="$2"
            shift 2
            ;;
        --)
            shift
            TRAIN_ARGS=("$@")
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument $1; put training arguments after --" >&2
            exit 2
            ;;
    esac
done

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "run id must contain only letters, numbers, dots, underscores, and hyphens" >&2
    exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "config does not exist: $CONFIG" >&2
    exit 1
fi
if ! python3 -m json.tool "$CONFIG" >/dev/null; then
    echo "config is not valid JSON: $CONFIG" >&2
    exit 1
fi
if ! command -v "${SBATCH_BIN:-sbatch}" >/dev/null 2>&1; then
    echo "sbatch is unavailable" >&2
    exit 1
fi

SUBMISSION_ROOT="${SUBMISSION_ROOT:-$REPOSITORY_ROOT/results/submitted-configs}"
SUBMISSION_DIRECTORY="$SUBMISSION_ROOT/$RESULT_SYSTEM/$RUN_ID"
mkdir -p "$SUBMISSION_ROOT/$RESULT_SYSTEM"
if ! mkdir "$SUBMISSION_DIRECTORY" 2>/dev/null; then
    echo "submission $RUN_ID already exists for $RESULT_SYSTEM; choose a different run id" >&2
    exit 1
fi
SNAPSHOT="$SUBMISSION_DIRECTORY/config.json"
cp "$CONFIG" "$SNAPSHOT"
chmod a-w "$SNAPSHOT"

JOB_SCRIPT="$SCRIPT_DIR/$JOB_NAME"
echo "run_id=$RUN_ID"
echo "source_config=$CONFIG"
echo "config_snapshot=$SNAPSHOT"
cd "$REPOSITORY_ROOT"
"${SBATCH_BIN:-sbatch}" \
    "--export=ALL,RUN_ID=$RUN_ID,CONFIG_SNAPSHOT=$SNAPSHOT" \
    "$JOB_SCRIPT" \
    "${TRAIN_ARGS[@]}"
