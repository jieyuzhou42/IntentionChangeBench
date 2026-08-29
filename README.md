# IntentionChangeBench

This repo has two separate pipelines:

- `src/simulation/`: generate gold trajectories.
- `src/eval/`: run the benchmark agent and score it.

Keep these separate. Simulation may expose gold user intention because it is creating the benchmark data. Eval must not expose gold intention to the agent; gold is used only after rollout for scoring.

## Bundled Environments

This repository includes the environment source snapshots used by the benchmark:

- `WebShop/`, snapshot `d29ba91`, based on [princeton-nlp/WebShop](https://github.com/princeton-nlp/WebShop).
- `TravelPlanner/`, snapshot `72d34bc`, based on [OSU-NLP-Group/TravelPlanner](https://github.com/OSU-NLP-Group/TravelPlanner).

Large downloaded datasets and generated search indexes are intentionally not stored in Git. Use the recovery steps below after cloning on a new machine.

## Directory Layout

```text
IntentionChangeBench/
  WebShop/
  TravelPlanner/
  scripts/
    setup_webshop_data.ps1

  src/
    common/
      execution_agent.py
      llm_clients.py

    domains/
      travelplanner/
        environment.py
        executor.py
        user_simulator.py
        run.py
      webshop/
        environment.py
        executor.py
        user_simulator.py
        run.py

    simulation/
      simulation/
        run_simulation.py
        base_user_simulator.py
        reranker.py
        runtime_logger.py

    eval/
      run_benchmark.py
      fixed_user_llm_executor.py
      benchmark_fixed_rollout.py
      evaluators/
        constraint_importance_eval.py
        runtime_logger.py

  data/
    simulation/
      simulated_dataset.json
      annotated_dataset.json

    eval/
      benchmark_eval.json
```

## Setup

Create `.env` from `.env.example` and fill in either the public OpenAI or
Azure OpenAI settings. For a normal OpenAI API key, use:

```text
LLM_PROVIDER=openai
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=your-model-id
```

The same benchmark prompt is sent when the configured model changes; only the
model selection changes. Model availability and supported request options
still depend on the account and model.

For a small DeepSeek test through its OpenAI-compatible Responses API, use:

```text
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_MODEL=deepseek-v4-flash
```

This repository also loads the ignored `.env.llm` profile for local testing.
Select `deepseek` or `openai` there, add the matching key and model, then run:

```powershell
python .\scripts\test_openai_api.py
```

To switch the same benchmark back to OpenAI, change `LLM_PROVIDER` to
`openai` and set `OPENAI_API_KEY` plus `OPENAI_MODEL`.

From the repo root, commands usually need WebShop on `PYTHONPATH`:

```powershell
$env:PYTHONPATH=(Resolve-Path .\WebShop).Path
```

Use the WebShop Python environment:

```powershell
.\.venv38-webshop\Scripts\python.exe ...
```

### Restore the full WebShop data

GitHub cannot store `WebShop/data/items_shuffle.json` because the file is about 5.1 GiB. The original WebShop project publishes the dataset through Google Drive. On a new Windows machine, download the same data with:

```powershell
.\scripts\setup_webshop_data.ps1 `
  -Python .\.venv38-webshop\Scripts\python.exe `
  -Dataset all
```

To also regenerate the full Lucene search resources and index:

```powershell
.\scripts\setup_webshop_data.ps1 `
  -Python .\.venv38-webshop\Scripts\python.exe `
  -Dataset all `
  -BuildIndex
```

The expected SHA-256 of the current `items_shuffle.json` is:

```text
2EF591D65DF3AF89E972AB72468EB82CBF124D876552D9F3678667EDD620A6C8
```

For full-data runs, set:

```powershell
$env:WEBSHOP_DATASET="all"
```

The full product file is required for `--webshop_num_products all`. The generated `WebShop/search_engine/resources/` and `WebShop/search_engine/indexes/` directories can be rebuilt and remain ignored by Git.

## Run Simulation

Simulation creates gold trajectories. It uses:

- `HumanSimulator` for user shifts and annotations.
- `WebShopExecutor` as the gold executor.
- BM25 search plus optional LLM reranking.
- Gold/current intention is allowed in this pipeline.

Smoke run:

```powershell
$env:PYTHONPATH=(Resolve-Path .\WebShop).Path
.\.venv38-webshop\Scripts\python.exe .\src\simulation\simulation\run_simulation.py `
  --output .\data\simulation\simulated_dataset.json `
  --domain webshop `
  --num_instances 2 `
  --max_turns 4 `
  --webshop_num_products 1000 `
  --parallelism 1
```

Fuller WebShop run:

```powershell
$env:PYTHONPATH=(Resolve-Path .\WebShop).Path
.\.venv38-webshop\Scripts\python.exe .\src\simulation\simulation\run_simulation.py `
  --output .\data\simulation\simulated_dataset.json `
  --domain webshop `
  --num_instances 20 `
  --max_turns 4 `
  --webshop_num_products 100000 `
  --parallelism 2 `
  --executor_type gold `
  --enable_reranking true
```

TravelPlanner uses the original ReAct action vocabulary rather than a one-shot
sole-planning shortcut:

```text
CitySearch / FlightSearch / AttractionSearch / AccommodationSearch /
RestaurantSearch / GoogleDistanceMatrix / NotebookWrite / Planner
```

TravelPlanner smoke run (30 internal steps is the original agent budget and is
the domain default):

```powershell
..\.venv38-webshop\Scripts\python.exe .\src\domains\travelplanner\run.py `
  --tasks_path .\data\simulation\_travelplanner_smoke_task.json `
  --output .\data\simulation\_travelplanner_sim_smoke.json `
  --num_instances 1 `
  --max_turns 4 `
  --parallelism 1
```

Each search produces an environment observation. `NotebookWrite` records that
observation, and `Planner` is the terminal action that constructs and evaluates
the itinerary. The full sequence is stored in each turn's `rollout_trace`.
TravelPlanner `env_feedback.search_results` groups the real search pages into
attractions, accommodations, restaurants, transportation, and cities. Each
page exposes up to 10 real items and explicitly reports when no results exist.
The submitted itinerary is kept separately; TravelPlanner does not use
`candidate_items`.

Important simulation args:

- `--num_instances`: number of tasks/goals to simulate.
- `--max_turns`: number of user turns after the initial turn.
- `--max_internal_steps`: max gold executor internal steps per turn (default 12
  for WebShop and 30 for TravelPlanner).
- `--webshop_num_products`: `100`, `1000`, `100000`, or `all`.
- `--parallelism`: number of instances to run concurrently.
- `--enable_reranking`: whether to rerank BM25 candidates with the LLM.

## Annotated Dataset

The benchmark input should live at:

```text
data/simulation/annotated_dataset.json
```

This file is the gold trajectory dataset used by eval. Each turn should include:

- `user_utterance`
- `gold_current_intention`
- `gold_delta`
- priority annotations such as `high`, `medium`, and `low`

## Run Benchmark / Test

Eval replays fixed user utterances and tests whether the agent can infer the user intention. It uses:

- `FixedUserLLMWebShopExecutor`
- default WebShop environment actions:
  - `search`
  - `click`
  - `buy`
  - `back_to_search`
  - `next_page`
  - `prev_page`
- agent-predicted current intention for environment feedback
- gold intention only after rollout for offline scoring

Gold intention is not passed into `env.step`.

Smoke benchmark:

```powershell
$env:PYTHONPATH=(Resolve-Path .\WebShop).Path
.\.venv38-webshop\Scripts\python.exe .\src\eval\run_benchmark.py `
  --gold_trajectory_path .\data\simulation\annotated_dataset.json `
  --output .\data\eval\_benchmark_eval_smoke.json `
  --num_instances 1 `
  --webshop_num_products 100000 `
  --parallelism 1 `
  --executor_type fixed_user
```

Full benchmark:

```powershell
$env:PYTHONPATH=(Resolve-Path .\WebShop).Path
.\.venv38-webshop\Scripts\python.exe .\src\eval\run_benchmark.py `
  --gold_trajectory_path .\data\simulation\annotated_dataset.json `
  --output .\data\eval\benchmark_eval.json `
  --webshop_num_products 100000 `
  --parallelism 2 `
  --executor_type fixed_user
```

Important eval args:

- `--gold_trajectory_path`: annotated/gold trajectory JSON.
- `--output`: benchmark output with eval scores.
- `--num_instances`: optional subset size.
- `--instance_ids`: comma-separated instance IDs to replay.
- `--max_turns`: optional max turn index to replay.
- `--max_internal_steps`: max agent WebShop actions per turn.
- `--parallelism`: number of benchmark instances to replay concurrently.

## Eval Scores

Each benchmark turn includes:

```json
"evaluation": {
  "state_understanding_eval": {},
  "action_selection_eval": {}
}
```

State understanding:

- `constraint_weighted_score`: did the agent predict the right constraint values?
- `priority_level_weighted_score`: did the agent assign constraints to the right high/medium/low importance level?
- `combined_weighted_score`: average of the two.

Action selection:

- `weighted_score`: selected item satisfaction against gold constraints.
- `selected_asin`: final selected item, if any.
- `per_constraint`: per-field credit and status.

Importance weights:

```text
high = 3
medium = 2
low = 1
```

## Aggregate Scores

Example aggregation command:

```powershell
python -c "import json; from collections import Counter; p='data/eval/benchmark_eval.json'; data=json.load(open(p,encoding='utf-8')); turns=[t for i in data for t in i.get('turns',[])]; avg=lambda xs: sum(xs)/len(xs) if xs else None; se=[t['evaluation']['state_understanding_eval'] for t in turns]; ae=[t['evaluation']['action_selection_eval'] for t in turns]; print('instances',len(data)); print('turns',len(turns)); print('state_constraint',avg([x.get('constraint_weighted_score',0) for x in se])); print('state_priority',avg([x.get('priority_level_weighted_score',0) for x in se])); print('state_combined',avg([x.get('combined_weighted_score',0) for x in se])); print('action',avg([x.get('weighted_score',0) for x in ae])); print('stop_reasons',dict(Counter(t.get('stop_reason') for t in turns)))"
```

## Prompt Logs

Prompts are written to:

```text
data/prompt_log.jsonl
```

The path is printed at the start of simulation and eval runs.
