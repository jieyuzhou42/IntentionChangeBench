#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
run_root="${2:-${WEBSHOP_FULL_RUN_ROOT:-}}"
repo_root="/fsx/sihengx/projects/baozai/IntentionChangeBench_pilot_20260903_160_170"
tasks_path="/fsx/sihengx/projects/baozai/IntentionChangeBench/data/simulation/webshop_diverse_350_formal.json"
python_bin="/fsx/sihengx/miniforge3/envs/intention-change-bench/bin/python"
if [[ -z "${run_root}" ]]; then
  echo "A run root argument or WEBSHOP_FULL_RUN_ROOT is required" >&2
  exit 2
fi

mkdir -p \
  "${run_root}/shards" \
  "${run_root}/logs" \
  "${run_root}/prompt_logs" \
  "${run_root}/nltk_data"

case "${mode}" in
  shard)
    shard_index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
    export PYTHONPATH="${repo_root}/WebShop:${repo_root}/src"
    export LLM_PROVIDER="bedrock"
    export BEDROCK_MODEL="us.openai.gpt-5.6-sol"
    export BEDROCK_REGION="us-west-2"
    export BEDROCK_MAX_TOKENS="4096"
    export BEDROCK_READ_TIMEOUT="300"
    export BEDROCK_MAX_RETRIES="8"
    export BEDROCK_RETRY_BACKOFF_SECONDS="2"
    export RERANKER_MAX_RETRIES="2"
    export WEBSHOP_ATTR_DATASET="all"
    export NLTK_DATA="${run_root}/nltk_data"
    export PROMPT_LOG_PATH="${run_root}/prompt_logs/shard_${shard_index}.jsonl"
    exec "${python_bin}" "${repo_root}/scripts/run_webshop_v7_shard.py" \
      --repo-root "${repo_root}" \
      --tasks-path "${tasks_path}" \
      --output-dir "${run_root}/shards" \
      --shard-index "${shard_index}" \
      --num-shards 50 \
      --base-seed 201
    ;;
  merge)
    exec "${python_bin}" "${repo_root}/scripts/merge_webshop_v7_shards.py" \
      --tasks-path "${tasks_path}" \
      --shards-dir "${run_root}/shards" \
      --num-shards 50 \
      --output "${run_root}/webshop_v7_full_350.json" \
      --manifest "${run_root}/webshop_v7_full_350.manifest.json"
    ;;
  *)
    echo "Usage: $0 {shard|merge}" >&2
    exit 2
    ;;
esac
