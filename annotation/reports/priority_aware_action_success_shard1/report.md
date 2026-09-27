# Priority-aware Action Success — shard 1

Semantic judgments: direct review by the assistant in this conversation, without external judge calls; counts and lexicographic comparison: Python. One count per human active constraint field; Optional is excluded and not exhaustively audited. No weighted sums.
A known undisclosed Must violation or invalid Agent action fails. Missing Gold and insufficient evidence are reported separately, never assumed successful. Unknown counts have lower/upper bounds; a verdict is given only if invariant to all completions. Missing/conflicting priority labels are not silently repaired. An untiered field satisfied by both sides cancels under every possible tier and does not block the comparison; shown counts omit these identical additions. Gold references marked confirmed=false are used provisionally and audited, not assumed optimal.

| Domain | Cases | Turns | Success | Failure | Unscorable | Success / scored |
|---|---:|---:|---:|---:|---:|---:|
| travelplanner | 6 | 32 | 12 | 11 | 9 | 52.17% |
| webshop | 8 | 40 | 20 | 5 | 15 | 80.00% |

WebShop source: bundle2.json filtered by the 8 original shard_001 instance IDs (40 turns). TravelPlanner: shard1, 6 cases / 32 turns; original Gold hash verified against output metadata.

| Domain / instance | Turn | Agent (M,P) | Gold (M,P) | Verdict | Reason |
|---|---:|---|---|---|---|
| travelplanner_test_0081 | 0 | (3, 0) | (3, 0) | success | preferred_at_least_gold |
| travelplanner_test_0081 | 1 | (2, 2–3) | (3, 3) | failure | undisclosed_must_violation |
| travelplanner_test_0081 | 2 | (1, 2) | (1, 3) | failure | preferred_fewer |
| travelplanner_test_0081 | 3 | (0, 0) | — | unscorable | missing_gold_action |
| travelplanner_test_0081 | 4 | (1, 0) | (1, 0) | success | preferred_at_least_gold |
| travelplanner_test_0081 | 5 | (0, 1) | (1, 0) | failure | undisclosed_must_violation |
| travelplanner_test_0126 | 0 | (1, 0) | (1, 0) | failure | invalid_agent_action |
| travelplanner_test_0126 | 1 | (3, 4) | (3, 4) | success | preferred_at_least_gold |
| travelplanner_test_0126 | 2 | (1, 1–2) | (1, 1–2) | unscorable | insufficient_evidence |
| travelplanner_test_0126 | 3 | (2, 0–1) | (2, 1–2) | unscorable | insufficient_evidence |
| travelplanner_test_0126 | 4 | (2–3, 0) | (2–3, 0) | unscorable | insufficient_evidence |
| travelplanner_test_0129 | 0 | (0, 0–1) | (0, 0) | failure | invalid_agent_action |
| travelplanner_test_0129 | 1 | (4, 3) | (1, 2) | success | must_more |
| travelplanner_test_0129 | 2 | (2, 3) | (2, 2) | success | preferred_at_least_gold |
| travelplanner_test_0129 | 3 | (1, 5) + common untiered | (1, 5) + common untiered | failure | undisclosed_must_violation |
| travelplanner_test_0129 | 4 | (3, 5) | (3, 5) | failure | undisclosed_must_violation |
| travelplanner_test_0129 | 5 | (2, 9) | (1, 9) | success | must_more |
| travelplanner_test_0144 | 0 | (1–2, 0) | (2, 0) | failure | invalid_agent_action |
| travelplanner_test_0144 | 1 | (4–5, 0) | (6, 0) | failure | undisclosed_must_violation |
| travelplanner_test_0144 | 2 | (3–4, 1) | (2–3, 0) | failure | invalid_agent_action |
| travelplanner_test_0357 | 0 | (3, 0) | (3, 0) | success | preferred_at_least_gold |
| travelplanner_test_0357 | 1 | (1, 3) | (2, 2) | unscorable | unlabeled_constraint |
| travelplanner_test_0357 | 2 | (3, 0–1) | (3, 1) | unscorable | unlabeled_constraint |
| travelplanner_test_0357 | 3 | (1, 4) | (1, 4) | unscorable | unlabeled_constraint |
| travelplanner_test_0357 | 4 | (1, 3) | (0, 3) | unscorable | unlabeled_constraint |
| travelplanner_test_0873 | 0 | (4, 0) | (4, 0) | failure | invalid_agent_action |
| travelplanner_test_0873 | 1 | (3, 1–2) + common untiered | (3, 1) + common untiered | success | comparison_robust_to_unknowns |
| travelplanner_test_0873 | 2 | (1–2, 3) + common untiered | (0, 3) + common untiered | success | comparison_robust_to_unknowns |
| travelplanner_test_0873 | 3 | (1–2, 1) + common untiered | (1, 1) + common untiered | success | comparison_robust_to_unknowns |
| travelplanner_test_0873 | 4 | (0, 2) + common untiered | (0, 2) + common untiered | success | preferred_at_least_gold |
| travelplanner_test_0873 | 5 | (1, 1) + common untiered | (1, 1) + common untiered | success | preferred_at_least_gold |
| travelplanner_test_0873 | 6 | (1, 2) | — | unscorable | missing_gold_action |
| webshop_goal_00000 | 0 | (1, 1) | (1, 1) | success | preferred_at_least_gold |
| webshop_goal_00000 | 1 | (1, 2) | (1, 2) | success | preferred_at_least_gold |
| webshop_goal_00000 | 2 | (2, 0) | (1, 0) | failure | undisclosed_must_violation |
| webshop_goal_00000 | 3 | (1, 1) | (3, 1) | failure | undisclosed_must_violation |
| webshop_goal_00318 | 0 | (1, 0) | (1, 0) | success | preferred_at_least_gold |
| webshop_goal_00318 | 1 | (3, 0) | (3, 0) | success | preferred_at_least_gold |
| webshop_goal_00318 | 2 | (1, 1–2) | (1, 0–1) | success | comparison_robust_to_unknowns |
| webshop_goal_01181 | 0 | (1, 0) | (1, 0) | success | preferred_at_least_gold |
| webshop_goal_01181 | 1 | (3–4, 5–8) | (3–4, 4–8) | unscorable | insufficient_evidence |
| webshop_goal_01181 | 2 | (2, 7–11) | (2, 7–10) | unscorable | insufficient_evidence |
| webshop_goal_01181 | 3 | (1–3, 6–11) | (2–3, 5–11) | unscorable | insufficient_evidence |
| webshop_goal_02449 | 0 | (1, 0) | (1, 0) | success | preferred_at_least_gold |
| webshop_goal_02449 | 1 | (1–2, 1–2) | — | unscorable | missing_gold_action |
| webshop_goal_02449 | 2 | (2, 2–3) | — | unscorable | missing_gold_action |
| webshop_goal_02449 | 3 | (1, 3–4) | (1, 4–5) | unscorable | insufficient_evidence |
| webshop_goal_02449 | 4 | (2, 5–6) | (2, 5–6) | unscorable | insufficient_evidence |
| webshop_goal_02449 | 5 | (2, 7) | (2, 7) | success | preferred_at_least_gold |
| webshop_goal_02702 | 0 | (0, 0) | (1, 0) | failure | must_fewer |
| webshop_goal_02702 | 1 | (1, 0) | (0, 0) | success | must_more |
| webshop_goal_02702 | 2 | (2, 0) | (1, 0) | success | must_more |
| webshop_goal_02702 | 3 | (2, 1) | — | unscorable | missing_gold_action |
| webshop_goal_02702 | 4 | (2, 1) | (0, 1) | success | must_more |
| webshop_goal_02702 | 5 | (1–2, 2) | (0–1, 1) | success | comparison_robust_to_unknowns |
| webshop_goal_05038 | 0 | (1, 0) | — | unscorable | missing_gold_action |
| webshop_goal_05038 | 1 | (2, 0) | — | unscorable | missing_gold_action |
| webshop_goal_05038 | 2 | (2, 1) | — | unscorable | missing_gold_action |
| webshop_goal_05038 | 3 | (2, 2) | — | unscorable | missing_gold_action |
| webshop_goal_05038 | 4 | (1–4, 1) | — | unscorable | missing_gold_action |
| webshop_goal_05038 | 5 | (2–3, 3) | — | unscorable | missing_gold_action |
| webshop_goal_05038 | 6 | (2, 2–4) | — | unscorable | missing_gold_action |
| webshop_goal_05121 | 0 | (6, 1) | (6, 1) | success | preferred_at_least_gold |
| webshop_goal_05121 | 1 | (6, 0) | (6, 0) | success | preferred_at_least_gold |
| webshop_goal_05121 | 2 | (2, 6) | (2, 6) | failure | undisclosed_must_violation |
| webshop_goal_05121 | 3 | (2, 5) | (2, 5) | success | preferred_at_least_gold |
| webshop_goal_05121 | 4 | (2, 5) | (2, 5) | success | preferred_at_least_gold |
| webshop_goal_06824 | 0 | (1, 0) | (1, 0) | success | preferred_at_least_gold |
| webshop_goal_06824 | 1 | (1, 0) | (1, 0) | success | preferred_at_least_gold |
| webshop_goal_06824 | 2 | (3, 0) | (3, 0) | success | preferred_at_least_gold |
| webshop_goal_06824 | 3 | (4, 1) | (4, 1) | success | preferred_at_least_gold |
| webshop_goal_06824 | 4 | (2, 1) | (2, 3) | failure | undisclosed_must_violation |
