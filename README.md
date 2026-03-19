# Intention Shift Benchmark Starter

This is a minimal Python framework for generating a multi-turn intention-shift benchmark with:

- 3 task types
- 10 base tasks per type
- latent state updates across turns
- natural-language user utterance realization
- JSON export for downstream execution / judge experiments

## Task types

- `scheduling`
- `retrieval_ranking`
- `transaction`

## Files

- `src/models.py`: data classes
- `src/base_tasks.py`: 30 hand-authored base tasks
- `src/generator.py`: shift sampling, state updates, utterance realization, dataset export
- `src/main.py`: CLI entry point
- `data/benchmark_dataset.json`: generated dataset

## Run

```bash
cd src
python main.py --output ../data/benchmark_dataset.json --seed 7 --num_shift_turns 3
```

## Output format

Each instance contains:

- `instance_id`
- `task_type`
- `subtype`
- `world_state`
- `turns`

Each turn contains:

- `turn_id`
- `user_utterance`
- `gold_delta`
- `gold_current_intention`
- `linguistic_style`
- `action_implication`

## Notes

This starter is intentionally simple. Likely next steps:

1. Replace random shift sampling with schema-constrained shift templates.
2. Add a compiler layer that derives structured state from dialogue history.
3. Add rule-based judges for state accuracy / tool-use accuracy / result correctness.
4. Add more human-like utterance realizations conditioned on conversation history.
