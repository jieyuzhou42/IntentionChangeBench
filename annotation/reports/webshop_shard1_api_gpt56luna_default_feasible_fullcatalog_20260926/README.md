# WebShop shard 1: default-feasible API evaluation

- Judge: gpt-5.6-luna (configured Responses API). Tested agent: gpt-5.6-sol.
- Source: annotation/output/webshop_output/gpt-5.6-sol__bundle2.json; gold from reviewed shard 1 ZIP (8 cases, 40 turns).
- Missing/null human world_feasibility defaults to true. Explicit false remains false; conflicts/invalid types remain errors.
- All 40 turns passed data preflight. Missing Gold products were recovered from original human-annotation observations, without changing annotations.
- 37/40 turns scored; 3 baseline schema failures remain unscored. These metrics cover only scored turns.
- Original dialogue and prior-turn predictions are retained. Existing src/eval prompts, validators and scorers are used.

| Model | Action score | Binary success | Must gate | Action coverage | Intention coverage |
|---|---:|---:|---:|---:|---:|
| gpt-5.6-sol | 50.72% | 45.95% | 54.05% | 37/40 | 37/40 |

Intention macro F1: 79.54%.

## Unscored turns

- webshop_goal_02702 turn 4: {'baseline': 'Judge failed: baseline criteria must cover every field exactly once'}
- webshop_goal_06824 turn 1: {'baseline': 'Judge failed: baseline criteria must cover every field exactly once'}
- webshop_goal_05038 turn 4: {'baseline': 'Judge failed: baseline criteria must cover every field exactly once'}

For webshop_goal_05038 turn 4, gold_delta adds kit_components including a power adapter, but gold_current_intention has only 9 fields and omits kit_components. The judge returns 10 criteria; validation correctly rejects the extra field. Other rejected baseline responses are saved under invalid_responses/. Intermediate rejected attempts can exist for ultimately successful turns; run_errors.json is the final error list.

## Files

- metrics.json, scored_rows.json, run_errors.json: final results.
- gold.json, catalog.json, trajectory.json: frozen inputs.
- baseline/, action/, intention/: validated cached judgments.
- run_manifest.json: model/source metadata.

Reproduce: python -u scripts/test_webshop_shard1_api.py. Uses .env.llm without printing credentials. Successful judgments are cached.

Verification: 12 shared Action scoring tests, 4 WebShop pipeline tests, and 1 TravelPlanner pipeline test passed.
