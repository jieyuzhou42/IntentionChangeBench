# TravelPlanner evaluation rules v2

Shared by every evaluated model. `## Baseline rules`, `## Action rules` and
`## Intention rules` are sent verbatim to the corresponding judge call;
`## Enforced in code` documents what Python computes and is not sent.
Change this file only together with a version bump in `RULES_VERSION`
(`src/eval/travelplanner_eval_v2.py`).

v3.0: World Feasibility is a human annotation; Python applies the Must count
gate and a capped 2:1 soft ratio against Gold. Budget, hotel and scope rules
below are unchanged. Disclosures remain diagnostic, not a satisfaction credit.

v2.2 (2026-09-24): the baseline writes frozen per-constraint criteria and the
action judge no longer sees the dialogue; baseline quotes must be verbatim
(checked in code). Disclosure is reported per violated Must, budget included.

Open decisions (defaults in force, pending team confirmation):
- Not-feasible turns: Agent must satisfy at least as many in-scope Must
  constraints as Gold. Matching the exact set of Gold violations is not required.
- House rules: a requirement to allow X fails only when the listing says "No X";
  "No visitors" does not by itself block "parties".
- Gold reference plans may be `confirmed=false`; they are still used for
  per-constraint comparison and the unchanged meal-slot reference.

## Baseline rules

You prepare a frozen audit baseline for ONE turn of a travel-planning dialogue.
It is shared by every evaluated agent, so it must not depend on any agent. You
see the dialogue so far, the gold constraints with tiers, the gold reference
plan (if any) and the candidate records the agents could choose from.

1. Gold atoms. Split every gold constraint into semantic atoms, one requirement
   per atom. Keep dates, day numbers, cities, travelers, negations and limits
   inside the atom. Copy the source field name into `source_field`. Never add
   information that the gold value does not state. A constraint that is already
   a single requirement becomes one atom.
2. Activity requirements. List every sightseeing, attraction, day-use or
   schedule requirement the user has explicitly stated up to this turn and not
   withdrawn (a later utterance replaces an earlier one). Do not infer
   requirements the user did not state; "passing through X" is not an exclusion.
   Every quote must be copied verbatim from the user's words (use "..." only
   to skip words); code rejects quotes that do not occur in the dialogue. kind is one of include,
   exclude, order, limit, free_time. Give the ISO date or null, the requirement,
   the turn it came from and a short quote.
3. Scope. Put a gold field in `out_of_scope` only if it is (a) an accessibility
   or mobility need of a traveler, or (b) the feasibility of moving around
   within a city (walking, transit, rideshare between stops). The candidate
   records cannot audit these. Nothing else is out of scope.
4. Criteria. For every in-scope gold field except budget, write `criteria`:
   one or two sentences stating exactly what a plan must do at this turn to
   satisfy the constraint, with scope resolved from the dialogue (which days,
   meals, cities, travelers) and the verbatim user quote it rests on (empty if
   the requirement comes only from the gold). Criteria restate the gold
   requirement; they never add, drop or relax it. When the dialogue conflicts
   with the gold, the criteria follow the gold and the conflict is reported as
   an annotation issue. These criteria are the only interpretation the action
   judge receives.
5. Gold plan audit. If a gold reference plan is given, judge every in-scope gold
   field except budget against it, with the action rules below: satisfied,
   violated or unknown, with a short evidence note. This establishes Gold's
   satisfied counts, NOT World Feasibility, which comes from human annotation.
   Budget, minimum nights and
   capacity of the gold plan are computed by code.
6. Annotation issues. Report conflicts between the gold constraints and the
   dialogue: a constraint the user withdrew, a value contradicting the latest
   utterance, a missing constraint the user clearly stated, an empty candidate
   pool. Describe them; do not correct the gold.

## Action rules

You audit ONE turn of ONE agent's travel plan against a frozen baseline. You do
not see the agent's intent prediction; judge only the plan.

1. Final plan. Start from the itinerary as written. Apply only explicit final
   revisions stated in the action (for example "replace X with Y", "switch the
   Day 2 hotel to Z"); record each in `applied_revisions` with its quote. Judge
   the revised plan as one whole: anything that conflicts with the final
   revision is judged on the final version. Mark every final_plan slot a
   revision changed with "revised": true.
2. Record mapping. For every non-empty transportation, meal and accommodation
   slot of the final plan, give `record_name`: the exact restaurant or listing
   name, or the flight number, from the candidate records; null when nothing
   matches. Code pre-matched names are supplied; keep them unless a revision
   changes the slot or the pre-match is clearly wrong. A slot may name several
   listings (a group split across rooms); join their exact names with " + ".
3. Facts come from the candidate records. Take prices, ratings, room types,
   house rules, minimum nights, maximum occupancy, cuisines and flight times
   from the records, never from the agent's wording or rationale. An entity
   absent from the records cannot satisfy a constraint that depends on its
   attributes: that constraint is violated, not unknown.
4. Criteria are the requirement. Each field to judge carries frozen criteria
   from the shared baseline; judge the plan against them. You do not see the
   dialogue, and must not reinterpret, add, drop or relax requirements. Per-city
   values such as {"Denver": 4} apply only to that city.
5. Activities. Judge activity, day and schedule constraints against their
   criteria and the baseline's activity requirements. Attractions and meals
   carry no clock times in this format: judge day placement and listed order,
   never answer unknown only because times are absent. Flight times are
   verifiable from the records.
6. people_number. Satisfied when the listings booked for every night together
   host that many travelers (sum of maximum occupancy) and nothing in the plan
   assumes another party size; violated when capacity falls short or the plan
   is built for another party size. Never unknown merely because the plan does
   not restate the head count.
7. Hotels. Judge room type, rating and house-rule constraints on the booked
   listing's own attributes, per city. Judge stay constraints (such as
   accommodation_stay) by the nights actually booked, one night per itinerary
   day with accommodation (the departure day needs none), against the requested
   dates, whatever the text claims. Minimum nights and maximum occupancy are a
   separate gate enforced by code; they do not change the verdict of any other
   constraint.
8. House rules. house_rules lists prohibitions. A requirement to allow an
   activity (parties, pets, smoking, visitors, children under 10) is violated
   only if the listing says "No <activity>". A requirement phrased "No <X>"
   needs the listing to prohibit X.
9. Meals. Meal-level constraints (restaurant rating, meal cost, cuisine,
   dining style, variety) apply to restaurant meals taken from the records,
   using Average Cost per person and Aggregate Rating. Self-catered, packed or
   empty slots are not evidence for or against them, unless the constraint
   itself requires a restaurant meal in that slot.
10. Budget is priced by code; do not judge it.
11. unknown is a last resort: only when the records genuinely lack the
    attribute that decides the constraint.
12. Disclosures. In `unmet_constraint_disclosures`, list every gold field that
    the action explicitly tells the user is not met (naming the requirement or
    its value), with the quote. Vague trade-off language does not count.
    Disclosure never changes action_status.
13. Annotation issues. Report data problems that prevent a fair audit, such as
    an empty candidate pool.

## Intention rules

You compare ONE agent's predicted intent at one turn with the frozen gold
atoms. You do not see any itinerary.

1. Split every predicted item into atoms with the same rule as the gold atoms:
   one requirement per atom, keeping dates, cities, travelers, negations and
   limits; never add information the item does not state. Give each atom the
   index of the predicted item it came from.
2. Match predicted atoms to gold atoms one-to-one: each gold atom and each
   predicted atom is used at most once. If two predicted atoms express the same
   gold atom, match the better one and leave the other unmatched. A renamed but
   clearly equivalent field may match.
3. value_match is true only if the atom's value is semantically correct for the
   gold atom's current value: Day 2 does not satisfy Day 1, a budget ceiling is
   not an exact spending target, an exclusion is not an inclusion.
   Also return scope_match as an explicit boolean. It is true only when the
   entity, traveler, city, day/time and applicability scope match the Gold atom.
   Same value for a different entity/scope is not a correct constraint.
   For unscoped requirements with no conflicting scope, return true when the
   Gold atom is matched; unmatched predictions must have scope_match=false.
   Never borrow a Gold entity/scope absent from the prediction unless the
   reference is unambiguously resolved by the supplied dialogue.
4. change_vs_previous compares each atom with the agent's own previous-turn
   prediction: unchanged if a previous item expresses the same requirement with
   the same value (renaming or rephrasing is not a change), changed if it
   expresses the same requirement with a different value, new if absent before.
   At the first turn every atom is new.
5. priority_change_vs_previous compares the priority of the same predicted
   requirement (including its entity/scope) to the agent's previous prediction:
   unchanged when the tier is identical, changed when it differs, new when no
   previous matching requirement exists. A value change alone does not imply a
   priority change. At the first turn every atom is new. Missing priority stays
   missing; never infer or fill in a tier from dialogue, Gold or importance.
   Match content without using priority to choose the Gold atom. Priority
   correctness is calculated in code from the original predicted item.

## Enforced in code

- Budget: TravelPlanner cost rule over the final plan's records (flights and
  meals per person, self-driving per 5, taxi per 4, accommodation per night
  per ceil(people / occupancy), several listings in one night priced one unit
  each). Meal coverage: a meal slot the gold plan fills with a priced
  restaurant must also be a priced restaurant in the agent plan, otherwise the
  budget is a coverage failure (violated). Slots the gold plan leaves empty may
  be omitted.
- Hotel gate: any listing booked for fewer consecutive nights than its minimum
  nights, or a night whose listings cannot host every traveler, fails the turn.
- Out-of-scope fields (baseline rule 3) are removed before scoring and counted.
- World Feasibility is read from the human annotation, never inferred from Gold
  failures. Missing/null human labels default to feasible=true.
- Must gate: feasible requires Ma=nM; not feasible requires Ma>=Mg. The existing
  Agent hotel gate remains enforced. Disclosure does not alter satisfaction.
- Sa=2Pa+Oa; Sg=2Pg+Og; soft=min(1,Sa/Sg), or 1 when Sg=0.
  Action score = (Must gate AND hotel valid) * soft, averaged over all scored
  turns including gate-failure zeros. No priority dominance beyond the Must gate.
- Missing Gold plan means assumed all-satisfied Gold for Ma/Mg and soft counts.
  Binary success then requires ALL active in-scope constraints satisfied;
  partial continuous credit remains possible after the gate passes.
- Unknown is not satisfied. API/schema failures are evaluation errors, not
  model constraint failures. In-scope fields need exactly one human priority.
- Preferred / Optional diagnostic rates remain available on Hard-Success turns.
- All Intention metrics are computed per turn, then averaged with equal turn
  weights. Micro metrics and the former pooled Priority Accuracy are removed.
- Content correct requires a one-to-one Gold match, value_match=true and
  scope_match=true. Conditional Priority Accuracy counts jointly correct
  content/scope/priority divided by content/scope-correct predictions. A turn
  with no correct content has an undefined conditional score and is excluded
  only from that metric, with coverage reported.
- Priority-aware Precision/Recall/F1 use jointly correct predictions as TP,
  all predicted atoms as the precision denominator and all current Gold atoms
  as the recall denominator. Missing/invalid predicted tiers cannot earn TP.
  Compute F1 per turn, not from the averaged Precision and Recall.
- Content Change uses add/override/relax. Delta Priority-aware metrics also
  include reprioritize and scope_correction. Expand old/new priority maps to
  affected current Gold fields; Gold current values and tiers are authoritative.
  First turns and turns without applicable current Gold changes are excluded
  with coverage. Removed/absent Gold constraints are not invented as targets.
  Include predictions linked to changed Gold even if unchanged by the agent;
  add spurious predicted content/priority changes to the precision denominator.
  Correct restatements of unchanged Gold requirements are not false positives.
- Every metric reports numerator, denominator and excluded count. API failures
  and invalid judge output are excluded and reported; missing Gold is scored
  with the explicit perfect-reference assumption above.
