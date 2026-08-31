# Simulation Pipeline

Purpose: generate gold trajectories.

Domain components are isolated under `src/domains/`:

```text
domains/
  travelplanner/
    environment.py
    executor.py
    user_simulator.py
  webshop/
    environment.py
    executor.py
    user_simulator.py
```

`simulation/simulation/run_simulation.py` contains only shared orchestration
and selects one complete domain bundle. Each domain owns its `run.py` entrypoint;
there are no domain-specific compatibility modules in the shared directory.

For TravelPlanner, generation follows the original multi-step tool workflow:
`CitySearch`, `FlightSearch`, `AttractionSearch`, `AccommodationSearch`,
`RestaurantSearch`, `GoogleDistanceMatrix`, `NotebookWrite`, and terminal
`Planner`. Reference information remains inside the environment and is exposed
only through the corresponding search observation. Each user turn starts with
a fresh Notebook, matching the original agent reset behavior.
The final user-simulator feedback groups every search page under
`search_results` by category and includes up to 10 real items per page. Empty
pages remain explicit. TravelPlanner does not expose generated plans as
candidate items.

TravelPlanner intentions additionally support named traveler entities. Shared
requirements remain in `constraints`; person-specific requirements live under
`entities.<entity_N>.constraints`. All person-level changes use the single
`category=entity` label plus the ordinary add/relax/override/reprioritize/
scope-correction operation. Membership changes keep `people_number`
synchronized, and stable opaque entity paths appear in `gold_delta`. Plans with person-specific
requirements use per-day `participant_assignments`, which the TravelPlanner
environment evaluates separately for each traveler. This entity schema is not
used by the WebShop domain.

Allowed:
- Expose and update gold user intention.
- Use the selected domain's `user_simulator.py` to create user shifts and annotations.
- Use the selected domain's `executor.py`; WebShop keeps BM25 search plus optional LLM reranking over the exposed intention state.

Not allowed:
- Import or run `eval.benchmark_fixed_rollout`.
- Treat this pipeline as the agent-understanding benchmark.
