#!/bin/bash
#SBATCH --job-name=sihengx-p4-dev
#SBATCH --partition=ml-r7i-48xlarge-us-west-2c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

set -euo pipefail

run_root="${REAL_ENV_RUN_ROOT:?REAL_ENV_RUN_ROOT is required}"
repo="${run_root}/repo"
python_bin="/fsx/sihengx/miniforge3/envs/intention-change-bench/bin/python"
output_name="${OUTPUT_NAME:?OUTPUT_NAME is required}"

cd "${repo}"
export PYTHONPATH="${repo}/src"

exec "${python_bin}" scripts/merge_collaborator_webshop_real_eval.py \
  --cases-dir "${run_root}/output/${output_name}_cases" \
  --output "${run_root}/output/${output_name}.json"
