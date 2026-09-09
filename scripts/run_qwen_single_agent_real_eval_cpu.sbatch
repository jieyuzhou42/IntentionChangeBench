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
agent_model="${QWEN_MODEL:-qwen.qwen3-235b-a22b-2507-v1:0}"
judge_model="${JUDGE_MODEL:-us.openai.gpt-5.6-sol}"

mkdir -p "${run_root}/output/qwen_cases" "${run_root}/logs"
cd "${repo}"

export PYTHONPATH="${repo}/WebShop:${repo}/src:${repo}/scripts"
export AWS_REGION="us-west-2"
export AWS_DEFAULT_REGION="us-west-2"
export WEBSHOP_DATASET="all"
export WEBSHOP_ATTR_DATASET="all"

pids=()
for case_index in $(seq 0 15); do
  "${python_bin}" scripts/run_qwen_single_agent_real_eval_case.py \
    --case-index "${case_index}" \
    --agent-model "${agent_model}" \
    --judge-model "${judge_model}" \
    --output "${run_root}/output/qwen_cases/case_${case_index}.json" \
    --timeout 600 \
    > "${run_root}/logs/qwen_case_${case_index}.log" 2>&1 &
  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
  wait "${pid}" || rc=1
done
if [[ "${rc}" -ne 0 ]]; then
  echo "One or more Qwen eval cases failed." >&2
  exit "${rc}"
fi

exec "${python_bin}" scripts/merge_single_agent_real_eval_cases.py \
  --cases-dir "${run_root}/output/qwen_cases" \
  --output "${run_root}/output/human_annotated_single_agent_real_eval_qwen_16.json"
