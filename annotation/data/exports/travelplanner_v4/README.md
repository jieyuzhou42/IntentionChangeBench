# TravelPlanner v4 annotated dataset

This directory contains 26 cases in three UTF-8 JSON arrays. Each array preserves
complete annotated case objects, including user turns, gold annotations and the
private reference data used by the search environment.

| File | Cases | Turns |
|---|---:|---:|
| `shard1_jieyu.json` | 6 | 32 |
| `shard_002_annotated(3).json` | 10 | 64 |
| `shard_003_annotated(2).json` | 10 | 64 |

`manifest.json` records per-file SHA-256 checksums, case IDs and turn counts.
The dataset JSON files are unchanged; the manifest describes the files present
in this export, without claiming unverified source provenance.

## Evaluation

Use the project evaluation pipeline and its dependencies. Present user turns
sequentially and let the evaluated agent perform its own searches. Keep gold,
stored actions/predictions, evaluation records and old rollout traces out of the
agent input. `world_state.reference_information` belongs to the tool environment.
Generate fresh agent predictions and actions instead of replaying saved actions.

The current prediction format is documented in `src/eval/INTENT_FORMAT.md`:
each intent item includes `field`, `value` and `priority`. Final actions should
select a concrete best-available solution and disclose unmet requirements.
