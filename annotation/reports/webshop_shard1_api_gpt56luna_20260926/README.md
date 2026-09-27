# WebShop shard 1: actual API evaluation

- Judge: `gpt-5.6-luna` via the configured OpenAI Responses API.
- Tested agent: `gpt-5.6-sol`, saved in `annotation/output/webshop_output/gpt-5.6-sol__bundle2.json`.
- Gold: reviewed ZIP, `shard1_5/shard_001_human_annotated.json` (8 cases, 40 turns).
- Uses existing `src/eval` v3 scoring and prompt/validation functions; source annotations and scoring rules unchanged.
- Full dialogue and previous-turn prediction are preserved for each evaluated turn.
- Product snapshot retains saved full candidate records, including descriptions, attributes, and options.

## Coverage and blockers

- 9/40 turns successfully scored for both Action and Intention. These are partial-subset metrics, not full-shard results.
- 30/40 turns have actual Gold actions but lack the required human `world_feasibility`; listed in `preflight_errors.json`.
- 1/40 turn (`webshop_goal_05038`, turn 4) repeatedly fails baseline schema validation. Gold current constraints contain 9 fields, but `gold_delta` adds `kit_components`; the judge emits 10 criteria, treating the delta field as active. Raw rejected responses are retained in `invalid_responses/`. No invalid response was accepted or scored.
- There are no remaining transport errors in the final run. Successful judgments were cached across retries.

## Results

| Model | Action score | Binary success | Must gate | Action coverage | Intention coverage |
|---|---:|---:|---:|---:|---:|
| gpt-5.6-sol | 49.07% | 44.44% | 55.56% | 9/40 | 9/40 |

| Intention metric | Value |
|---|---:|
| Macro precision | 80.88% |
| Macro recall | 88.20% |
| Macro F1 | 82.88% |
| Exact turn | 11.11% |
| Priority accuracy | 24.59% |

## Scored turns

| Instance | Turn | Action score | Success | Intention F1 |
|---|---:|---:|---:|---:|
| webshop_goal_02702 | 3 | 0.0000 | False | 0.7143 |
| webshop_goal_05038 | 0 | 1.0000 | True | 1.0000 |
| webshop_goal_05038 | 1 | 1.0000 | True | 0.6667 |
| webshop_goal_05038 | 2 | 1.0000 | True | 0.8333 |
| webshop_goal_05038 | 3 | 1.0000 | True | 0.8571 |
| webshop_goal_05038 | 5 | 0.0000 | False | 0.8421 |
| webshop_goal_05038 | 6 | 0.4167 | False | 0.8571 |
| webshop_goal_02449 | 1 | 0.0000 | False | 0.8000 |
| webshop_goal_02449 | 2 | 0.0000 | False | 0.8889 |

## Reproduce

Run `python -u scripts/test_webshop_shard1_api.py` from the repository root with the configured `.env.llm` and network access. Existing successful judgments are reused. See `run_manifest.json`, `metrics.json`, `scored_rows.json`, and `run_errors.json` for machine-readable outputs.

The Intention nested metric excluded counts are relative to the already-filtered scored subset in the current summarizer; use top-level coverage (9/40) for shard-wide coverage.
