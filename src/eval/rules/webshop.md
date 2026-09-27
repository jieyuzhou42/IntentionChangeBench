# WebShop evaluation rules v1.0

These are per-constraint evidence rules, identical for Agent and actual Gold.
The judge does not compute scores, priorities or World Feasibility.

1. A product without evidence supporting a requested constraint FAILS that
   constraint. Return violated with a missing_evidence explanation, not a
   speculative satisfied. Legacy unknown verdicts also receive zero credit.
   Agent claims, common knowledge and plausible product properties are not evidence.
2. When there is no Gold action, Python assumes the annotator's reference
   satisfies ALL active constraints. Agent must satisfy ALL constraints for
   binary Success; any missing/violated constraint means Failure. Continuous
   credit is separate: a Must-gated partial soft score is not binary Success.
3. Judge the selected ASIN AND selected options. An available size/color is not
   a selected size/color. Do not invent a default when the environment does not
   record it. Inspect full descriptions, bullets and attributes, not only titles.
4. Each frozen constraint is one scoring unit. For AND requirements every
   component needs support; for OR requirements one alternative suffices.
   Do not split, merge, relax or remove fields. Report ambiguous booleans,
   duplicate requirements and stale annotations as annotation issues at baseline.
5. Black does not prove matte black; metal does not prove steel. Hooks do not
   prove hook-and-loop closure. An included outdoor antenna does not prove an
   included indoor antenna. Compatibility does not prove a component is included.
   A product explicitly labeled glam fails an exclusion of glam even if it is
   also described as modern. Never invent attribute values or units.
6. A negative ingredient requirement needs supporting evidence of absence;
   lack of a keyword alone is not proof. Follow the baseline's interpretation
   of ambiguous fields such as artificial_ingredients=true.
7. Use the selected product/variant price and the criterion's exact comparator
   (< versus <=); distinguish per-item prices, pack contents and quantity.
8. Disclosures are exact quotes from the selected action rationale, reported
   separately. They do not change whether a constraint is satisfied. Python
   applies the human feasible/not-feasible Must count gate and soft 2:1 ratio.
