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
