# Evaluation intention prediction

New evaluation runs store `agent_intention_prediction` as:

```json
{
  "intent": [
    {"field": "return_flight", "value": "F3623037", "priority": "must_have"},
    {"field": "budget_max", "value": 2000, "priority": "preferred"},
    {"field": "day_2_required_activity", "value": "Cleveland Cultural Gardens", "priority": "must_have"}
  ]
}
```

Each item contains its own priority: `must_have`, `preferred`, or `optional`.
There is no separate priority list, `relation`, or `scope`. Relevant context
(days, travelers, inclusion/exclusion, limits) is retained in the field or value.
Distinct items sharing a field are preserved. Exact duplicate items are rejected.

The action schema, original gold and saved historical runs are unchanged.
Blind prediction, fixed-search, single-agent TravelPlanner and fixed-user WebShop
entry points request and validate this same structure. Their outer response keys
(`current_intention_understanding` or `predicted_current_intention`) are retained.

Semantic judging receives the complete intent list and original gold. Deterministic
state evaluation compares field, value and per-item tier, without collapsing repeated
fields. Old high/medium/low gold is mapped to must_have/preferred/optional for scoring.
The deterministic legacy bridge covers top-level gold; paraphrased fields, compound
prose and legacy entity constraints require semantic judging.

Legacy environments receive an internal compatibility view with supported unique
fields and the full intent array. Repeated/contextual fields are not flattened. This
view is not saved as the agent prediction, and heuristic environment feedback does
not replace semantic evaluation.

## Concrete best-available actions

Evaluation prompts require a final, concrete choice even if no candidate satisfies
every user requirement. The agent chooses trade-offs using its inferred priorities,
keeps the intention prediction unchanged, and explains unmet requirements in its
action rationale. Search-capable agents should research alternatives before settling.
Hotel minimum nights and occupancy remain feasibility conditions, not preferences.
Travel plans must select actual lodging, transport and meals rather than placeholders
or unchosen alternatives. Unknown prices are disclosed rather than invented.

Judging still counts the selected plan's unmet constraints. Disclosing a compromise
does not earn compliance credit. A truly empty catalog must not cause fabricated
choices; it is reported explicitly. Fixed-candidate WebShop normalization rejects
`no_match` when real candidate IDs are present. Natural-language completeness of
travel plans is a prompted requirement and is assessed by the judge, not guaranteed
by the structural JSON parser.
