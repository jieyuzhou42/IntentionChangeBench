# Gold-authority retry: WebShop shard 1

GPT-5.6-Luna re-evaluated the 3 turns that failed baseline validation in the preceding run. All 3 now have valid Action and Intention scores, with no final errors.

Active fields, values and priorities come exclusively from normalized gold_current_intention. Baseline prompts omit gold_delta; dialogue may clarify meaning but cannot add/remove/replace/relax Gold constraints. Extra agent predictions retain existing Intention precision treatment. No source annotations were edited.

| Instance | Turn | Action score | Intention F1 |
|---|---:|---:|---:|
| webshop_goal_02702 | 4 | 1.0000 | 0.5882 |
| webshop_goal_06824 | 1 | 1.0000 | 0.6000 |
| webshop_goal_05038 | 4 | 0.0000 | 0.6316 |

This report covers only the 3 retried turns. The other 37 turns were not rerun under the revised prompt; no mixed-run aggregate is presented.

Validation: 5 WebShop pipeline tests and 12 shared Action scoring tests passed.

Reproduce:

```powershell
python -u scripts/test_webshop_shard1_api.py --out annotation/reports/webshop_shard1_gold_authority_retry_20260927 --turn webshop_goal_02702:4 --turn webshop_goal_06824:1 --turn webshop_goal_05038:4
```
