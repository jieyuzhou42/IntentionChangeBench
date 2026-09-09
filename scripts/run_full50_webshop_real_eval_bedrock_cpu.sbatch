#!/bin/bash
#SBATCH --job-name=sihengx-p4-dev
#SBATCH --partition=ml-r7i-48xlarge-us-west-2c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=512G
#SBATCH --time=06:00:00

set -euo pipefail

export LLM_PROVIDER="bedrock"
export BEDROCK_MODEL="${AGENT_MODEL:?AGENT_MODEL is required}"
export BEDROCK_REGION="us-west-2"
export BEDROCK_MAX_TOKENS="8192"
export BEDROCK_READ_TIMEOUT="600"
export BEDROCK_MAX_RETRIES="8"
export BEDROCK_RETRY_BACKOFF_SECONDS="2"

exec bash scripts/run_full50_webshop_real_eval_cpu.sbatch
