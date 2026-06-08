# Simulation Pipeline

Purpose: generate gold trajectories.

Allowed:
- Expose and update gold user intention.
- Use `HumanSimulator` to create user shifts and annotations.
- Use the gold executor (`WebShopExecutor`): BM25 search plus optional LLM reranking over the exposed intention state.

Not allowed:
- Import or run `eval.benchmark_fixed_rollout`.
- Treat this pipeline as the agent-understanding benchmark.
