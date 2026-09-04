# Human Annotation

This directory keeps the human-annotation workflow separate from simulation and evaluation:

- `replay_server.py`: browser-based rollout review and editing tool.
- `data/annotated_dataset.json`: reviewed gold trajectories consumed by evaluation.
- `data/replay_image_cache.json`: generated WebShop image cache (ignored by Git).
- `tests/`: annotation-tool tests.

Run the annotation server from the repository root:

```powershell
..\.venv38-webshop\Scripts\python.exe .\annotation\replay_server.py `
  --dataset .\data\simulation\simulated_dataset.json `
  --host 127.0.0.1 `
  --port 7860
```

The source dataset is read-only. Without `--output`, edits are saved as
`annotation/data/<dataset_stem>_annotated.json`; an existing output is resumed.

Annotators can also:

- confirm a domain-specific `gold_action` for every turn;
- select a WebShop product and its exact options from the candidate list;
- review and edit TravelPlanner selections in day-by-day cards;
- insert blank turns or delete turns (each instance keeps at least one turn).
- delete an entire trajectory from the annotation output (the file keeps at least one trajectory).

Edits and turn-structure changes stay in the browser while moving between turns.
`Save Trajectory` atomically writes every turn in the current instance. Moving to
another instance saves the current trajectory first.

Constraint state is cumulative across a trajectory. Adding a constraint carries it
into later turns with the same priority. Value changes and removals also propagate
while later turns still match the old value; an explicit later override is preserved.
The annotation UI labels stored `high`, `medium`, and `low` priorities as
`Must-have`, `Preferred`, and `Optional`, respectively.
For selected WebShop tasks, authoritative `selection_metadata.attributes` and
`selection_metadata.options` are restored when an older rollout omitted them.

For WebShop reranked rollouts, the page displays 15 candidates: the first 10 keep
their reranker metadata and the next 5 come from the recorded raw results. Products
are hydrated on demand from the WebShop catalog through `data/replay_item_cache.json`,
including options, descriptions, highlights, brand/color, rating, availability,
seller, and structured product information when present.

Confirmed WebShop actions require a product ASIN. Confirmed TravelPlanner actions
require every day to specify the city, transportation, meals, attraction, and
accommodation; use `-` when an item is intentionally not needed.

## Firebase collaboration site

`annotation/firebase` contains a Firebase Hosting + Firestore version of the
WebShop annotator. It serves source shards as read-only static files and saves each
edited trajectory independently in the `webshop_annotations` collection. Visitors
are signed in anonymously in the background, so collaborators do not need an email
address or an interactive login while direct unauthenticated database requests stay
blocked. Trajectory turns are stored as `turns_json` so catalog metadata with keys
that Firestore cannot represent natively remains lossless.
Build the generated site data from the repository root with:

```powershell
..\.venv38-webshop\Scripts\python.exe .\annotation\firebase\build_site.py
```

The current deployment range is shard 6 through shard 20 (150 trajectories).
