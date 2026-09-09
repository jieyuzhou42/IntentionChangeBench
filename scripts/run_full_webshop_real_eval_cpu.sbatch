#!/bin/bash
#SBATCH --job-name=sihengx-p4-dev
#SBATCH --partition=ml-r7i-48xlarge-us-west-2c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=512G
#SBATCH --time=06:00:00

set -euo pipefail

run_root="${REAL_ENV_RUN_ROOT:?REAL_ENV_RUN_ROOT is required}"
repo="${run_root}/repo"
python_bin="/fsx/sihengx/miniforge3/envs/intention-change-bench/bin/python"
agent_model="${AGENT_MODEL:?AGENT_MODEL is required}"
judge_model="${JUDGE_MODEL:-us.openai.gpt-5.6-sol}"
output_name="${OUTPUT_NAME:?OUTPUT_NAME is required}"
case_dir="${run_root}/output/${output_name}_cases"
parallelism="${CASE_PARALLELISM:-20}"

mkdir -p "${case_dir}" "${run_root}/logs"
cd "${repo}"

export PYTHONPATH="${repo}/WebShop:${repo}/src:${repo}/scripts"
export AWS_REGION="us-west-2"
export AWS_DEFAULT_REGION="us-west-2"
export WEBSHOP_DATASET="all"
export WEBSHOP_ATTR_DATASET="all"

for ((start=0; start<50; start+=parallelism)); do
  end=$((start + parallelism - 1))
  if ((end > 49)); then end=49; fi
  pids=()
  for ((case_index=start; case_index<=end; case_index++)); do
    "${python_bin}" scripts/run_full_webshop_real_eval_case.py \
      --case-index "${case_index}" \
      --agent-model "${agent_model}" \
      --judge-model "${judge_model}" \
      --output "${case_dir}/case_${case_index}.json" \
      --timeout 600 \
      > "${run_root}/logs/${output_name}_case_${case_index}.log" 2>&1 &
    pids+=("$!")
  done
  rc=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || rc=1
  done
  if [[ "${rc}" -ne 0 ]]; then
    echo "One or more WebShop cases failed in wave ${start}-${end}." >&2
    exit "${rc}"
  fi
done

exec "${python_bin}" scripts/merge_full_webshop_real_eval.py \
  --cases-dir "${case_dir}" \
  --output "${run_root}/output/${output_name}.json"
