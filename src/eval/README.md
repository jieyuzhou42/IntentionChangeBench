# Eval Pipeline

Purpose: benchmark whether the agent understands user intention from fixed user utterances and WebShop observations.

Allowed:
- Replay fixed `user_utterance` values from a gold trajectory.
- Let `eval/fixed_user_llm_executor.py` interact with the default WebShop environment using `search`, `click`, `buy`, `back_to_search`, `next_page`, and `prev_page`.
- Use `gold_current_intention` only after rollout for offline scoring.

Not allowed:
- Pass `gold_current_intention` into the executor or `env.step`.
- Import the simulation rollout (`simulation.simulation.run_simulation.execute_turn`).
- Use the gold BM25/direct executor for benchmark action selection.

## Saved-trajectory Action evaluation (v3)

`prepare -> run -> summarize` is retained. The shared scorer is
`action_scoring.py`; baseline normalization and identities are in `baseline.py`.
TravelPlanner keeps `travelplanner_eval_v2.py` and its original command for
compatibility. WebShop uses `webshop_eval.py`, `webshop_checks.py`, and
`rules/webshop.md`. Both are accessible through `scripts/run_eval.py`.

### Scoring

Only in-scope human active constraints count. Every such field needs exactly
one human Must/Preferred/Optional tier; missing/conflicting priorities are data
errors, not silently excluded. Existing TravelPlanner out-of-scope rules remain.

- Human **feasible**: Must gate is `Ma == nM`.
- Human **not feasible**: Must gate is `Ma >= Mg`. It compares counts, not the
  identity of sacrificed constraints. Disclosures remain reported separately.
- `Sa = 2*Pa + Oa`, `Sg = 2*Pg + Og`.
- `soft = min(1, Sa/Sg)`, or `1` if `Sg == 0`.
- Final continuous score is `gate_pass * soft`. TravelPlanner's existing hotel
  validity gate is retained in `gate_pass`; WebShop checks actual selection validity.
- Missing Gold **action** means assumed all-satisfied Gold for every tier. In
  this case binary `action_success` requires ALL in-scope constraints satisfied;
  partial continuous credit after the Must gate does not mean binary Success.
- With actual Gold, binary success means the gate passes and soft credit is 1.
- Missing product support gives no satisfaction credit. Legacy `unknown` also
  gives zero credit, while judge/API/data errors are reported as unscored.
- Average over all scored turns, including Must failures scored 0. Action and
  Intention have independent coverage/denominators.

TravelPlanner's budget computation (including `unpriced` handling), Gold meal
coverage checks, minimum-night/capacity checks and out-of-scope rules are unchanged.
World Feasibility is no longer inferred from Gold constraint violations.

### Human annotation and frozen baseline

Preferred source:

```json
{"gold_action": {"world_feasibility": {"feasible": false}}}
```

The same `world_feasibility` boolean/object is also accepted directly on the
turn or in `annotation_review`. Conflicting flags are rejected. Missing or null
human flags default to **feasible=true**, with or without a Gold action; this
requires all Must constraints to be satisfied. Explicit **false** still uses
the Gold Must count gate. This default applies to both domains and the shared
scorer. Use a new output directory when rerunning older baselines. Missing Gold **intention**
is an error; no constraints or priorities are inferred from the tested model.

The baseline freezes human constraints, priorities, per-field criteria,
Intention gold atoms, Gold verdicts, human feasibility and source hashes.
All tested models use this same artifact. Changing source data, rules or
baseline invalidates caches. Old v2 caches cannot be used with v3 scoring;
use a new output directory. No source annotations are modified by evaluation.

For WebShop, normalized `gold_current_intention.constraints` is the sole
authority for active scoring constraints, including supported entity constraints.
Neither `gold_delta`, dialogue, nor agent predictions can add/remove Gold fields
or override their values/priorities. The baseline judge does not receive
`gold_delta`; it is retained for change scoring against existing Gold atoms.
Extra agent predictions retain the existing Intention precision treatment and
never become additional Action requirements. Baseline field coverage remains
strictly validated; missing Gold fields are not silently discarded.

### TravelPlanner commands

The existing manifest-based loader and flags are retained:

```powershell
python scripts/run_eval.py prepare --domain travelplanner --run-dir RUN_DIR --gold-dir GOLD_DIR --out OUT_DIR
python scripts/run_eval.py run --domain travelplanner --run-dir RUN_DIR --gold-dir GOLD_DIR --out OUT_DIR
python scripts/run_eval.py summarize --domain travelplanner --run-dir RUN_DIR --gold-dir GOLD_DIR --out OUT_DIR
```

The old `scripts/run_travelplanner_eval_v2.py` command is equivalent, without
`--domain`. Existing rollout source hashes must still match the annotated shard.

### WebShop commands

`--gold` is a JSON array of annotated cases (`instance_id`, `turns`, each turn's
`gold_current_intention`, optional `gold_action`, human World Feasibility).
`--catalog` is a frozen product array or ASIN-to-product map. Products have
`asin`, `title`, `price`, description/bullets/attributes and `options` mapping
option names to supported values. Supply the full records, not display-only
projections; an unresolved selected ASIN is a data error, not evidence that
the product lacks a requested attribute.

```powershell
python scripts/run_eval.py prepare --domain webshop --gold GOLD.json --catalog CATALOG.json --out OUT_DIR
python scripts/run_eval.py run --domain webshop --gold GOLD.json --catalog CATALOG.json --trajectory model-a=TRAJECTORY_A.json --trajectory model-b=TRAJECTORY_B.json --out OUT_DIR
python scripts/run_eval.py summarize --domain webshop --gold GOLD.json --catalog CATALOG.json --trajectory model-a=TRAJECTORY_A.json --trajectory model-b=TRAJECTORY_B.json --out OUT_DIR
```

Trajectories support the saved `trajectories: [{instance_id, turns}]` format or
flat `rows` with `instance_id` and `turn_id`. Final selection is read from saved
environment feedback/action evidence/action payload, with actual selected
options from the payload, feedback or final rollout trace. The Agent's
rationale cannot supply unsupported product facts. Use `--instances` to limit
all stages to the same cases. Missing expected tested turns appear in coverage.

Both runners support `--dump-prompts DIR` for inspecting prompts without API
calls. `run` requires a prepared baseline. Online judging uses
`OPENROUTER_API_KEY`; `summarize` requires no API key. No model is invoked by
the pure scorer. Outputs are `scored_rows.json`, `metrics.json`, `tables.md`.
WebShop judge errors are also recorded in `run_errors.json`.

### Verification without external APIs

```powershell
python -B -m unittest discover -s tests -p test_action_scoring_v3.py
python -B -m unittest discover -s tests -p test_webshop_eval_pipeline.py
python -B -m unittest discover -s tests -p test_travelplanner_pipeline_v3.py
```

The pipeline tests use synthetic saved trajectories and a local fake judge.
They test baseline reuse, stale-cache rejection, independent coverage, missing
Gold, selection options and the complete prepare/run/summarize flow.
