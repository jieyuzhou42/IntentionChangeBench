#!/bin/bash
#SBATCH --job-name=sihengx-p4-dev
#SBATCH --partition=ml-r7i-48xlarge-us-west-2c
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/fsx/sihengx/projects/baozai/IntentionChangeBench/data/simulation/webshop_v2_runtime/logs/sihengx-p4-dev-repair-%j.log

set -euo pipefail

REPO=/fsx/sihengx/projects/baozai/IntentionChangeBench
cd "$REPO"
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
  scripts/repair_webshop_v2_rerank.py
