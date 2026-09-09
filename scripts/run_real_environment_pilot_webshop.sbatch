#!/bin/bash
#SBATCH --job-name=sihengx-p4-dev
#SBATCH --partition=ml-r7i-48xlarge-us-west-2c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail

run_root="${REAL_ENV_RUN_ROOT:?REAL_ENV_RUN_ROOT is required}"
repo="${run_root}/repo"
python_bin="/fsx/sihengx/miniforge3/envs/intention-change-bench/bin/python"

mkdir -p "${run_root}/output" "${run_root}/logs"
cd "${repo}"

export PYTHONPATH="${repo}/WebShop:${repo}/src"
export LLM_PROVIDER="bedrock"
export BEDROCK_MODEL="us.openai.gpt-5.6-sol"
export BEDROCK_REGION="us-west-2"
export BEDROCK_MAX_TOKENS="8192"
export BEDROCK_READ_TIMEOUT="600"
export BEDROCK_MAX_RETRIES="8"
export BEDROCK_RETRY_BACKOFF_SECONDS="2"
export WEBSHOP_DATASET="all"
export WEBSHOP_ATTR_DATASET="all"

exec "${python_bin}" scripts/run_real_environment_pilot_eval.py \
  --domain webshop \
  --webshop_max_internal_steps 12 \
  --output "${run_root}/output/webshop_real_env.json" \
  --timeout 600
