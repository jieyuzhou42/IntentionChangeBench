#!/bin/bash
#SBATCH --job-name=shx-tests
#SBATCH --partition=ml-r7i-48xlarge-us-west-2c
#SBATCH --nodes=5
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=/fsx/sihengx/projects/baozai/IntentionChangeBench/data/simulation/webshop_v3_runtime/logs/sihengx-p4-dev-%j.log

set -euo pipefail

REPO=/fsx/sihengx/projects/baozai/IntentionChangeBench
cd "$REPO"
mkdir -p data/simulation/webshop_v3_runtime/logs

export PYTHONPATH="$REPO/WebShop:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export LLM_PROVIDER=bedrock
export BEDROCK_MODEL=us.openai.gpt-5.6-sol
export BEDROCK_REGION=us-west-2
export BEDROCK_MAX_TOKENS=4096
export BEDROCK_READ_TIMEOUT=300
export BEDROCK_MAX_RETRIES=8
export BEDROCK_RETRY_BACKOFF_SECONDS=2
export WEBSHOP_ATTR_DATASET=all

/fsx/sihengx/miniforge3/envs/intention-change-bench/bin/python \
  scripts/run_webshop_v3_multinode.py \
  --source data/simulation/webshop_diverse_350_formal.json \
  --output data/simulation/webshop_v3_350_formal.json \
  --runtime-dir data/simulation/webshop_v3_runtime \
  --node-groups 5 \
  --shards-per-node 10 \
  --shard-size 7 \
  --max-worker-attempts 3
