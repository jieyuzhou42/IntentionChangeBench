# Script entry points

## Current saved-trajectory evaluation

| Entry | Purpose |
|---|---|
| `run_eval.py` | Shared `prepare`, `run`, `summarize` interface for WebShop and TravelPlanner. |
| `run_travelplanner_eval_v2.py` | TravelPlanner implementation behind `run_eval.py`; its filename is retained for CLI compatibility. |
| `test_webshop_shard1_api.py` | Focused WebShop shard 1 API check, including `--turn` selection with full dialogue context. Reads local `.env.llm`; requires saved trajectories. |
| `mine_travelplanner_judge_cases.py` | Judge diagnostics; the current TravelPlanner runner also imports its database verdict helper. |

Scoring, prompt rules, validation, and metric definitions live in
[`src/eval`](../src/eval/README.md), not in one-off rescoring scripts.
Current Intention metrics use equal turn weights and include Conditional
Priority Accuracy, Priority-aware F1 and Delta Priority-aware F1.

## Other retained workflows

- `run_*`, `merge_*`, `finalize_*`, `.sbatch`, and sync scripts support agent
  rollouts and cluster execution; they are not interchangeable with offline eval.
- Task selection, constraint extraction, priority classification, context
  normalization and WebShop setup/repair scripts support data preparation.
- `judge_travelplanner_v4_trajectories.py` and
  `report_travelplanner_v4_eval.py` support historical reports. The latter also
  supplies helpers used by judge diagnostics. They do not implement the current
  Intention metric definitions.

## Retired experimental evaluators

The following obsolete one-off implementations and their dedicated tests were
removed. Their saved historical reports/data remain available under
`annotation/reports`; those results are not relabeled as current metrics.

- `eval_priority_aware_action_success.py`
- `eval_shard1_dominance_weighted.py`
- `eval_shard1_must_gated_soft.py`
- `eval_shard1_soft_2to1.py`
- `eval_shard1_soft_gold_ratio.py`
- `eval_webshop_feasible_soft_value.py`
- `eval_webshop_missing_gold_perfect.py`
- `eval_webshop_shards1_5_soft_2to1.py`
- `eval_webshop_strict_guideline.py`
- `eval_webshop_three_tier_direct_review.py`
- `judge_travelplanner_v4.py` (separate from the retained `_trajectories.py` entry)
- `prepare_webshop_shards1_5_review.py`
- `record_shard1_direct_review.py`
- `record_webshop_shards2_5_direct.py`
- `rescore_webshop_v1_soft_2to1.py`
- `show_webshop_direct_review.py`

The unused `merge_real_environment_pilot_outputs.py.orig` backup was also removed;
the real `.py` entry used by SLURM remains.
