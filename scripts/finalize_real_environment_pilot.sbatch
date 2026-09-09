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

cd "${repo}"
export PYTHONPATH="${repo}/src"

exec "${python_bin}" scripts/merge_real_environment_pilot_outputs.py \
  --webshop "${run_root}/output/webshop_real_env.json" \
  --travelplanner "${run_root}/input/travelplanner_real_env.json" \
  --output "${run_root}/output/human_annotated_real_env_pilot_16.json"
