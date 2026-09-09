#!/usr/bin/env bash
set -uo pipefail

MODE="${1:-shard}"
ROUND="${2:-1}"
REPO="${REPO:-/fsx/sihengx/projects/baozai/IntentionChangeBench}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
PYTHON="${PYTHON:-$REPO/.venv-travelplanner/bin/python}"
TASKS_PATH="${TASKS_PATH:-$REPO/data/travelplanner/short_horizon_250_multi_360_tasks.json}"
BASELINE_PATH="${BASELINE_PATH:-$REPO/data/simulation/_travelplanner_limit1_entity_multi_api_20260831.json}"
OUTPUT_PATH="${OUTPUT_PATH:?OUTPUT_PATH is required}"
NUM_SHARDS="${NUM_SHARDS:-50}"
PARALLELISM="${PARALLELISM:-8}"
PARTITION="${PARTITION:-ml-r7i-48xlarge-us-west-2c}"
MAX_FINALIZE_ROUNDS="${MAX_FINALIZE_ROUNDS:-50}"
JOB_NAME="${JOB_NAME:-sihengx-p4-dev}"
SHARD_CPUS="${SHARD_CPUS:-8}"
SHARD_MEM="${SHARD_MEM:-16G}"
DRIVER="$REPO/scripts/run_travelplanner_v2_shard.py"

mkdir -p "$RUN_DIR/logs"

export LLM_PROVIDER=bedrock
export BEDROCK_MODEL="${BEDROCK_MODEL:-us.openai.gpt-5.6-sol}"
export BEDROCK_REGION="${BEDROCK_REGION:-us-west-2}"
export BEDROCK_MAX_TOKENS="${BEDROCK_MAX_TOKENS:-8192}"
export BEDROCK_READ_TIMEOUT="${BEDROCK_READ_TIMEOUT:-600}"
export BEDROCK_MAX_RETRIES="${BEDROCK_MAX_RETRIES:-10}"
export BEDROCK_RETRY_BACKOFF_SECONDS="${BEDROCK_RETRY_BACKOFF_SECONDS:-3}"
export PROMPT_LOG_ENABLED=0
export PYTHONUNBUFFERED=1

common_args=(
  --repo "$REPO"
  --python "$PYTHON"
  --tasks-path "$TASKS_PATH"
  --baseline-path "$BASELINE_PATH"
  --run-dir "$RUN_DIR"
  --num-shards "$NUM_SHARDS"
  --seed 20260905
  --max-turns 6
  --max-internal-steps 50
  --parallelism "$PARALLELISM"
)

submit_finalize() {
  local dependency="$1"
  local next_round="$2"
  sbatch \
    --parsable \
    --dependency="afterany:$dependency" \
    --partition="$PARTITION" \
    --cpus-per-task=2 \
    --mem=8G \
    --time=24:00:00 \
    --job-name="$JOB_NAME" \
    --export="ALL,REPO=$REPO,RUN_DIR=$RUN_DIR,PYTHON=$PYTHON,OUTPUT_PATH=$OUTPUT_PATH,NUM_SHARDS=$NUM_SHARDS,PARALLELISM=$PARALLELISM,PARTITION=$PARTITION,MAX_FINALIZE_ROUNDS=$MAX_FINALIZE_ROUNDS,JOB_NAME=$JOB_NAME,SHARD_CPUS=$SHARD_CPUS,SHARD_MEM=$SHARD_MEM,TASKS_PATH=$TASKS_PATH,BASELINE_PATH=$BASELINE_PATH" \
    --output="$RUN_DIR/logs/finalize-%j.log" \
    "$0" finalize "$next_round"
}

if [[ "$MODE" == "shard" ]]; then
  : "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required for shard mode}"
  exec "$PYTHON" "$DRIVER" run-shard \
    "${common_args[@]}" \
    --shard-index "$SLURM_ARRAY_TASK_ID" \
    --max-attempts 3
fi

if [[ "$MODE" != "finalize" ]]; then
  echo "Unknown mode: $MODE" >&2
  exit 64
fi

set +e
"$PYTHON" "$DRIVER" finalize \
  "${common_args[@]}" \
  --output "$OUTPUT_PATH"
finalize_status=$?
set -e

if [[ "$finalize_status" -eq 0 ]]; then
  echo "TravelPlanner formal run is complete."
  exit 0
fi

missing_shards="$(
  "$PYTHON" - "$RUN_DIR/status.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    status = json.load(handle)
print(",".join(str(index) for index in status.get("missing_shards", [])))
PY
)"
if [[ -z "$missing_shards" ]]; then
  echo "Finalize failed without a recoverable missing-shard list." >&2
  exit "$finalize_status"
fi
if (( ROUND >= MAX_FINALIZE_ROUNDS )); then
  date -u +%FT%TZ > "$RUN_DIR/FAILED"
  echo "Stopped after $ROUND finalize rounds; missing shards: $missing_shards" >&2
  exit 2
fi

retry_job="$(
  sbatch \
    --parsable \
    --array="$missing_shards" \
    --partition="$PARTITION" \
    --cpus-per-task="$SHARD_CPUS" \
    --mem="$SHARD_MEM" \
    --time=24:00:00 \
    --job-name="$JOB_NAME" \
    --export="ALL,REPO=$REPO,RUN_DIR=$RUN_DIR,PYTHON=$PYTHON,OUTPUT_PATH=$OUTPUT_PATH,NUM_SHARDS=$NUM_SHARDS,PARALLELISM=$PARALLELISM,PARTITION=$PARTITION,MAX_FINALIZE_ROUNDS=$MAX_FINALIZE_ROUNDS,JOB_NAME=$JOB_NAME,SHARD_CPUS=$SHARD_CPUS,SHARD_MEM=$SHARD_MEM,TASKS_PATH=$TASKS_PATH,BASELINE_PATH=$BASELINE_PATH" \
    --output="$RUN_DIR/logs/retry-${ROUND}-%A_%a.log" \
    "$0" shard
)"
next_finalize="$(submit_finalize "$retry_job" "$((ROUND + 1))")"
echo "Resubmitted missing shards as $retry_job; finalizer is $next_finalize"
