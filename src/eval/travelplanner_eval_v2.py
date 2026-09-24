"""TravelPlanner evaluation v2: frozen baseline, per-turn judges, scoring in code.

Three judge calls, all governed by ``rules/travelplanner_v2.md``:

* baseline   once per (instance, turn), shared by every model: gold atoms,
             activity requirements, out-of-scope fields, the gold plan's
             constraint verdicts (feasibility) and annotation issues;
* action     once per (model, turn): final plan with record mapping, explicit
             revisions, per-constraint verdicts with evidence, disclosures;
* intention  once per (model, turn): predicted atoms matched one-to-one to the
             gold atoms, with change status against the previous turn.

Judges return judgments and evidence only. Budget, hotel gate, meal coverage,
Hard Success and every score are computed here.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from eval.travelplanner_checks import (
    FLIGHT_NUMBER,
    MEALS,
    _is_empty_slot,
    _match_all,
    _reference_entries,
    _text,
    hotel_validity,
    meal_coverage_gaps,
    trip_cost,
)

RULES_VERSION = "travelplanner-rules-v2.2"
REPO_ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = Path(__file__).with_name("rules") / "travelplanner_v2.md"
CALIBRATION_PATH = REPO_ROOT / "annotation" / "data" / "travelplanner_judge_calibration_v1.json"
AUDIT_PATH = REPO_ROOT / "annotation" / "data" / "exports" / "travelplanner_v4" / "gold_audit_v1.json"

GOLD_TIER = {"high": "must_have", "medium": "preferred", "low": "optional"}
TIERS = ("must_have", "preferred", "optional")
ACTION_STATUSES = {"satisfied", "violated", "unknown"}
CHANGE_STATUSES = {"new", "changed", "unchanged"}
VALUE_CHANGE_OPS = {"add", "override", "relax"}
PLAN_SLOTS = ("transportation",) + MEALS + ("accommodation",)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


# --------------------------------------------------------------------------- rules, audit, few-shot


def load_rules(path: Path = RULES_PATH) -> Dict[str, str]:
    """Rule sections keyed by their ``## `` heading."""
    sections: Dict[str, str] = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = ""
        elif current:
            sections[current] += line + "\n"
    return {name: body.strip() for name, body in sections.items()}


def load_audit(path: Path = AUDIT_PATH) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["entries"] if path.exists() else []


def apply_gold_audit(
    instance_id: str, turn_id: int, gold: Dict[str, Any], audit: Sequence[Dict[str, Any]]
) -> Tuple[Dict[str, Any], List[str]]:
    """Apply audit entries with status 'applied'; the original gold is untouched."""
    gold = copy.deepcopy(gold)
    applied = []
    for entry in audit:
        if entry.get("status") != "applied" or entry.get("instance_id") != instance_id:
            continue
        if turn_id not in entry.get("turns", []):
            continue
        change = entry.get("apply") or {}
        constraints = gold.setdefault("constraints", {})
        for field in change.get("remove_fields", []):
            constraints.pop(field, None)
            for level in (gold.get("priority") or {}).values():
                if isinstance(level, list) and field in level:
                    level.remove(field)
        for field, value in (change.get("set_fields") or {}).items():
            constraints[field] = value
        applied.append(entry["id"])
    return gold, applied


def load_calibration(path: Path = CALIBRATION_PATH) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def fewshot_block(stage: str, exclude_instance: str, cases: Sequence[Dict[str, Any]]) -> str:
    """Worked examples for ``stage``, leaving out cases from the judged instance."""
    lines = []
    for case in cases:
        if case.get("stage") != stage or not case.get("use_as_fewshot"):
            continue
        if case["source"]["instance_id"] == exclude_instance:
            continue
        if stage == "baseline":
            verdict = {"out_of_scope": [{"field": case["field"], "reason": case["lesson"]}]}
        else:
            verdict = {"gold_field": case["field"], "action_status": case["expected"]["action_status"]}
            if case["expected"].get("unmet_disclosed"):
                verdict["disclosure_quote"] = case["expected"].get("disclosure_quote", "")
        lines.append(
            f"- Lesson: {case['lesson']}\n"
            f"  Constraint: {case['field']} = {_json(case['gold_value'])} ({case['tier']})\n"
            f"  Evidence: {case['evidence']}\n"
            f"  Correct: {_json(verdict)}"
        )
    if not lines:
        return ""
    return (
        "WORKED EXAMPLES (excerpts from other trips; apply the reasoning, not the facts):\n"
        + "\n".join(lines)
    )


# --------------------------------------------------------------------------- shared helpers


def gold_constraints(gold: Dict[str, Any]) -> Dict[str, Any]:
    return {str(k): v for k, v in (gold.get("constraints") or {}).items() if v is not None}


def gold_tiers(gold: Dict[str, Any]) -> Dict[str, str]:
    priority = gold.get("priority") or {}
    return {str(f): GOLD_TIER[level] for level in GOLD_TIER for f in priority.get(level) or []}


def dialogue_so_far(turns: Sequence[Dict[str, Any]], turn_index: int) -> List[Dict[str, Any]]:
    return [
        {"turn": int(t.get("turn_id", i)), "user": t.get("user_utterance")}
        for i, t in enumerate(turns[: turn_index + 1])
    ]


def _normalize(text: Any) -> str:
    return re.sub(r"[\W_]+", " ", str(text or "").lower()).strip()


def quote_in_dialogue(quote: Any, dialogue: Sequence[Dict[str, Any]]) -> bool:
    """True when every part of ``quote`` (split at an ellipsis) occurs in the user turns."""
    text = _normalize(" ".join(str(t.get("user") or "") for t in dialogue))
    parts = [_normalize(p) for p in re.split(r"\.\.\.|…", str(quote or ""))]
    parts = [p for p in parts if p]
    return bool(parts) and all(p in text for p in parts)


def intent_items(prediction: Any) -> List[Dict[str, Any]]:
    if isinstance(prediction, dict) and isinstance(prediction.get("intent"), list):
        return [item for item in prediction["intent"] if isinstance(item, dict)]
    return []


def code_matches(itinerary: Any, reference: Any) -> List[Dict[str, Any]]:
    """Exact database records for each plan slot, as code can resolve them."""
    entries = _reference_entries(reference)
    flights = {
        str(e["Flight Number"]).upper()
        for value in (reference or {}).values() if isinstance(value, list)
        for e in value if isinstance(e, dict) and e.get("Flight Number")
    }
    out = []
    for index, day in enumerate(itinerary if isinstance(itinerary, list) else []):
        if not isinstance(day, dict):
            continue
        for slot in PLAN_SLOTS:
            text = _text(day.get(slot))
            if _is_empty_slot(text):
                continue
            if slot == "transportation":
                numbers = [n.upper() for n in FLIGHT_NUMBER.findall(text) if n.upper() in flights]
                name = " + ".join(numbers) or None
            elif slot == "accommodation":
                name = " + ".join(r["NAME"] for r in _match_all(text, entries["stays"], "NAME")) or None
            else:
                found = _match_all(text, entries["restaurants"], "Name")
                name = found[0]["Name"] if len(found) == 1 else None
            out.append({"day": index + 1, "slot": slot, "record_name": name})
    return out


# --------------------------------------------------------------------------- baseline


def build_baseline_prompt(payload: Dict[str, Any], rules: Dict[str, str], fewshot: str) -> str:
    schema = {
        "gold_atoms": [{"atom_id": "g1", "source_field": "exact gold field", "value": "one requirement"}],
        "activity_requirements": [
            {"kind": "include | exclude | order | limit | free_time", "date": "YYYY-MM-DD or null",
             "requirement": "...", "source_turn": 0, "quote": "..."}
        ],
        "out_of_scope": [{"field": "exact gold field", "reason": "..."}],
        "constraint_criteria": [{"gold_field": "exact gold field", "criteria": "what a plan must do at this turn",
                                 "quote": "verbatim user words, or empty"}],
        "gold_plan_judgments": [{"gold_field": "exact gold field", "status": "satisfied | violated | unknown", "evidence": "..."}],
        "annotation_issues": ["..."],
    }
    return "\n\n".join(
        part for part in (
            rules["Baseline rules"],
            "Action rules used for the gold plan audit (judge the gold plan against the criteria you "
            "write; the note that the action judge does not see the dialogue does not apply here):\n"
            + rules["Action rules"],
            fewshot,
            "Return exactly one JSON object:\n" + json.dumps(schema, indent=2),
            "source_field must be copied exactly from gold_constraints[].field; never invent field names. "
            "Cover every gold field with at least one atom. Give constraint_criteria for every in-scope "
            "field except budget. Every quote must be copied verbatim from dialogue_so_far. "
            "If gold_reference_plan is null, return an empty gold_plan_judgments list; otherwise judge "
            "every in-scope field except budget.",
            "PAYLOAD:\n" + _json(payload),
        ) if part
    )


def validate_baseline(
    raw: Dict[str, Any], gold: Dict[str, Any], has_plan: bool, dialogue: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    fields = set(gold_constraints(gold))
    atoms = raw.get("gold_atoms")
    if not isinstance(atoms, list) or not atoms:
        raise ValueError("baseline: gold_atoms missing")
    covered = {str(a.get("source_field")) for a in atoms if isinstance(a, dict)}
    if covered - fields:
        raise ValueError(f"baseline: atoms name unknown fields {sorted(covered - fields)}")
    if fields - covered:
        raise ValueError(f"baseline: fields without atoms {sorted(fields - covered)}")
    ids = [str(a.get("atom_id")) for a in atoms]
    if len(set(ids)) != len(ids):
        raise ValueError("baseline: duplicate atom ids")
    out_of_scope = {str(o.get("field")) for o in raw.get("out_of_scope") or [] if isinstance(o, dict)}
    if out_of_scope - fields:
        raise ValueError(f"baseline: out_of_scope names unknown fields {sorted(out_of_scope - fields)}")
    in_scope = fields - out_of_scope - {"budget"}
    criteria = {str(c.get("gold_field")): c for c in raw.get("constraint_criteria") or [] if isinstance(c, dict)}
    if set(criteria) != in_scope:
        raise ValueError(
            f"baseline: criteria missing={sorted(in_scope - set(criteria))} extra={sorted(set(criteria) - in_scope)}"
        )
    for c in criteria.values():
        if not str(c.get("criteria") or "").strip():
            raise ValueError(f"baseline: empty criteria for {c.get('gold_field')}")
        if str(c.get("quote") or "").strip() and not quote_in_dialogue(c["quote"], dialogue):
            raise ValueError(f"baseline: criteria quote not in dialogue for {c.get('gold_field')}: {c['quote'][:80]}")
    for req in raw.get("activity_requirements") or []:
        if not quote_in_dialogue((req or {}).get("quote"), dialogue):
            raise ValueError(f"baseline: activity quote not in dialogue: {str((req or {}).get('quote'))[:80]}")
    judged = {str(j.get("gold_field")): j for j in raw.get("gold_plan_judgments") or [] if isinstance(j, dict)}
    if has_plan:
        expected = fields - out_of_scope - {"budget"}
        if set(judged) != expected:
            raise ValueError(
                f"baseline: gold plan judgments missing={sorted(expected - set(judged))} extra={sorted(set(judged) - expected)}"
            )
        for j in judged.values():
            if j.get("status") not in ACTION_STATUSES:
                raise ValueError(f"baseline: bad gold plan status {j.get('status')}")
    return raw


# --------------------------------------------------------------------------- action


def build_action_prompt(payload: Dict[str, Any], rules: Dict[str, str], fewshot: str) -> str:
    schema = {
        "final_plan": [{"day": 1, "slot": "transportation | breakfast | lunch | dinner | accommodation",
                        "text": "final slot text", "record_name": "exact record name(s) or flight number, or null",
                        "revised": False}],
        "applied_revisions": [{"quote": "...", "effect": "..."}],
        "constraint_judgments": [{"gold_field": "exact gold field", "action_status": "satisfied | violated | unknown",
                                  "evidence": "record fact or quote", "note": "brief reason"}],
        "unmet_constraint_disclosures": [{"gold_field": "exact gold field", "quote": "..."}],
        "annotation_issues": ["..."],
    }
    return "\n\n".join(
        part for part in (
            rules["Action rules"],
            fewshot,
            "Return exactly one JSON object:\n" + json.dumps(schema, indent=2),
            "Judge every field listed in fields_to_judge exactly once, against its frozen criteria.",
            "PAYLOAD:\n" + _json(payload),
        ) if part
    )


def validate_action(raw: Dict[str, Any], fields_to_judge: Set[str]) -> Dict[str, Any]:
    rows = raw.get("constraint_judgments")
    if not isinstance(rows, list):
        raise ValueError("action: constraint_judgments missing")
    by_field: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        field = str((row or {}).get("gold_field"))
        if field in by_field:
            raise ValueError(f"action: duplicate judgment for {field}")
        if (row or {}).get("action_status") not in ACTION_STATUSES:
            raise ValueError(f"action: bad status {row.get('action_status')} for {field}")
        by_field[field] = row
    if set(by_field) != fields_to_judge:
        raise ValueError(
            f"action: missing={sorted(fields_to_judge - set(by_field))} extra={sorted(set(by_field) - fields_to_judge)}"
        )
    if not isinstance(raw.get("final_plan", []), list):
        raise ValueError("action: final_plan must be a list")
    return raw


def final_itinerary(itinerary: Any, final_plan: Sequence[Dict[str, Any]], revised_days: Set[Tuple[int, str]], reference: Any) -> List[Dict[str, Any]]:
    """The plan code checks run on: written itinerary, plus judge-resolved slots.

    Code-matched slots keep the written text. A slot the judge revised, or one
    code could not resolve, takes the judge's record name when that name is a
    real record; otherwise the written text stays.
    """
    days = [copy.deepcopy(d) if isinstance(d, dict) else {} for d in (itinerary if isinstance(itinerary, list) else [])]
    matches = {(m["day"], m["slot"]): m["record_name"] for m in code_matches(itinerary, reference)}
    entries = _reference_entries(reference)
    for item in final_plan or []:
        try:
            day, slot = int(item.get("day")), str(item.get("slot"))
        except (TypeError, ValueError):
            continue
        if slot not in PLAN_SLOTS or day < 1:
            continue
        while len(days) < day:
            days.append({})
        name = item.get("record_name")
        revised = (day, slot) in revised_days
        if not revised and matches.get((day, slot)):
            continue
        if revised and not name:
            days[day - 1][slot] = item.get("text") or "-"
            continue
        if name:
            pool = entries["stays"] if slot == "accommodation" else entries["restaurants"] if slot in MEALS else None
            real = slot == "transportation" or (pool is not None and _match_all(str(name), pool, "NAME" if slot == "accommodation" else "Name"))
            if real:
                days[day - 1][slot] = str(name)
    return days


# --------------------------------------------------------------------------- intention


def build_intention_prompt(payload: Dict[str, Any], rules: Dict[str, str]) -> str:
    schema = {
        "pred_atoms": [{"source_index": 0, "field": "...", "value": "one requirement",
                        "gold_atom_id": "g1 or null", "value_match": True,
                        "change_vs_previous": "new | changed | unchanged"}]
    }
    return "\n\n".join(
        (
            rules["Intention rules"],
            "Return exactly one JSON object:\n" + json.dumps(schema, indent=2),
            "Every predicted item index must yield at least one atom.",
            "PAYLOAD:\n" + _json(payload),
        )
    )


def validate_intention(raw: Dict[str, Any], n_items: int, gold_ids: Set[str], first_turn: bool) -> Dict[str, Any]:
    atoms = raw.get("pred_atoms")
    if not isinstance(atoms, list):
        raise ValueError("intention: pred_atoms missing")
    sources, used = set(), set()
    for atom in atoms:
        if str(atom.get("gold_atom_id")).strip().lower() in {"", "null", "none"}:
            atom["gold_atom_id"] = None
        try:
            sources.add(int(atom.get("source_index")))
        except (TypeError, ValueError):
            raise ValueError("intention: bad source_index")
        gid = atom.get("gold_atom_id")
        if gid is not None:
            gid = str(gid)
            if gid not in gold_ids:
                raise ValueError(f"intention: unknown gold atom {gid}")
            if gid in used:
                raise ValueError(f"intention: gold atom {gid} matched twice")
            used.add(gid)
        if atom.get("change_vs_previous") not in CHANGE_STATUSES:
            raise ValueError(f"intention: bad change status {atom.get('change_vs_previous')}")
    if sources != set(range(n_items)):
        raise ValueError(f"intention: atoms cover items {sorted(sources)}, expected {n_items}")
    if first_turn:
        for atom in atoms:
            atom["change_vs_previous"] = "new"
    return raw


# --------------------------------------------------------------------------- scoring


def _ratio(num: float, den: float) -> Optional[float]:
    return num / den if den else None


def _f1(p: Optional[float], r: Optional[float]) -> float:
    p, r = p or 0.0, r or 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def budget_verdict(itinerary: Any, gold_plan: Any, reference: Any, budget: Any, people: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(budget, (int, float)):
        return None
    cost = trip_cost(itinerary, reference, people or 1)
    gaps = meal_coverage_gaps(itinerary, gold_plan, reference) if gold_plan else []
    status = "satisfied" if cost["total"] <= budget and not gaps else "violated"
    return {"status": status, "total": cost["total"], "budget": budget, "unpriced": cost["unpriced"], "coverage_gaps": gaps}


def feasibility(baseline: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, Any]:
    """Which in-scope Must constraints the gold plan itself violates."""
    code = baseline.get("gold_plan_code") or {}
    if not baseline.get("has_gold_plan"):
        return {"status": "gold_plan_missing", "violated_musts": []}
    tiers = gold_tiers(gold)
    out = set(baseline["out_of_scope_fields"])
    violated = {
        j["gold_field"] for j in baseline["judge"].get("gold_plan_judgments") or []
        if j.get("status") == "violated" and tiers.get(j["gold_field"]) == "must_have" and j["gold_field"] not in out
    }
    if (code.get("budget") or {}).get("status") == "violated" and tiers.get("budget") == "must_have":
        violated.add("budget")
    return {"status": "not_feasible" if violated else "feasible", "violated_musts": sorted(violated),
            "gold_hotel_valid": (code.get("hotel") or {}).get("valid")}


def score_action(
    *,
    gold: Dict[str, Any],
    baseline: Dict[str, Any],
    judgment: Dict[str, Any],
    itinerary: Any,
    reference: Any,
    gold_plan: Any,
    people: Any,
) -> Dict[str, Any]:
    constraints = gold_constraints(gold)
    tiers = gold_tiers(gold)
    out = set(baseline["out_of_scope_fields"])
    in_scope = [f for f in constraints if f not in out]
    revised = {
        (int(p.get("day")), str(p.get("slot")))
        for p in judgment.get("final_plan") or []
        if isinstance(p, dict) and p.get("revised") is True and str(p.get("day", "")).isdigit()
    }
    plan = final_itinerary(itinerary, judgment.get("final_plan") or [], revised, reference)
    status = {j["gold_field"]: j["action_status"] for j in judgment["constraint_judgments"]}
    budget = budget_verdict(plan, gold_plan, reference, constraints.get("budget"), people)
    if budget is not None and "budget" in in_scope:
        status["budget"] = budget["status"]
    hotel = hotel_validity(plan, reference, people)
    disclosed = {str(d.get("gold_field")) for d in judgment.get("unmet_constraint_disclosures") or [] if isinstance(d, dict)}
    musts = [f for f in in_scope if tiers.get(f) == "must_have"]
    violated_musts = [f for f in musts if status.get(f) == "violated"]
    undisclosed = [f for f in violated_musts if f not in disclosed]
    feas = feasibility(baseline, gold)

    if feas["status"] == "not_feasible":
        gold_bad = set(feas["violated_musts"])
        must_ok = all(
            status.get(f) == "satisfied" or (f in gold_bad and status.get(f) == "violated" and f in disclosed)
            for f in musts
        )
    else:
        must_ok = all(status.get(f) == "satisfied" for f in musts)
    hard_success = must_ok and hotel["valid"]

    by_tier: Dict[str, Dict[str, int]] = {}
    for f in in_scope:
        tier = tiers.get(f, "entity")
        bucket = by_tier.setdefault(tier, {"satisfied": 0, "violated": 0, "unknown": 0})
        bucket[status.get(f, "unknown")] += 1
    return {
        "hard_success": hard_success,
        "strict_success": all(status.get(f) == "satisfied" for f in musts),
        "feasibility": feas,
        "hotel": hotel,
        "budget": budget,
        "musts": len(musts),
        "violated_musts": violated_musts,
        "undisclosed_violated_musts": undisclosed,
        "disclosed": sorted(disclosed),
        "out_of_scope": sorted(out & set(constraints)),
        "status": status,
        "by_tier": by_tier,
        "revisions": len(judgment.get("applied_revisions") or []),
    }


def score_intention(
    *,
    gold: Dict[str, Any],
    gold_delta: Dict[str, Any],
    baseline: Dict[str, Any],
    judgment: Dict[str, Any],
    items: Sequence[Dict[str, Any]],
    first_turn: bool,
) -> Dict[str, Any]:
    tiers = gold_tiers(gold)
    gold_atoms = {str(a["atom_id"]): a for a in baseline["judge"]["gold_atoms"]}
    atoms = judgment["pred_atoms"]
    correct = [a for a in atoms if a.get("gold_atom_id") is not None and a.get("value_match")]
    precision = _ratio(len(correct), len(atoms)) or 0.0
    recall = _ratio(len(correct), len(gold_atoms)) or 0.0

    def tier_of(atom: Dict[str, Any]) -> Optional[str]:
        index = int(atom["source_index"])
        return items[index].get("priority") if index < len(items) else None

    matched = {str(a["gold_atom_id"]): a for a in atoms if a.get("gold_atom_id") is not None}
    tiered = [gid for gid, g in gold_atoms.items() if tiers.get(str(g["source_field"])) in TIERS]
    tier_pairs = [(tiers[str(gold_atoms[gid]["source_field"])], tier_of(matched[gid])) for gid in tiered if gid in matched]

    changed_fields = {f for f, d in (gold_delta or {}).items() if isinstance(d, dict) and d.get("op") in VALUE_CHANGE_OPS}
    changed_gold = {gid for gid, g in gold_atoms.items() if str(g["source_field"]) in changed_fields}
    change = None
    if not first_turn and changed_gold:
        claimed = []
        for atom in atoms:
            if atom.get("change_vs_previous") == "unchanged":
                continue
            gid = atom.get("gold_atom_id")
            if gid is not None and str(gid) not in changed_gold and atom.get("value_match"):
                continue  # correct restatement of an unchanged requirement
            claimed.append(gid is not None and str(gid) in changed_gold and bool(atom.get("value_match")))
        caught = sum(1 for gid in changed_gold if gid in matched and matched[gid].get("value_match"))
        p = _ratio(sum(claimed), len(claimed)) or 0.0
        r = caught / len(changed_gold)
        change = {"gold": len(changed_gold), "caught": caught, "claimed": len(claimed),
                  "claimed_correct": sum(claimed), "precision": p, "recall": r, "f1": _f1(p, r)}
    return {
        "atoms_gold": len(gold_atoms),
        "atoms_pred": len(atoms),
        "correct": len(correct),
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "turn_exact": len(correct) == len(atoms) == len(gold_atoms),
        "tier_pairs": tier_pairs,
        "priority_turn_exact": len(tier_pairs) == len(tiered) and all(g == p for g, p in tier_pairs),
        "change": change,
    }


def summarize(rows: Sequence[Dict[str, Any]], shards: Iterable[str]) -> Dict[str, Any]:
    """Turn-macro metrics with numerator / denominator / excluded counts."""
    def metric(values: List[float], excluded: int = 0) -> Dict[str, Any]:
        return {"value": _ratio(sum(values), len(values)), "numerator": sum(values),
                "denominator": len(values), "excluded": excluded}

    scored = [r for r in rows if r.get("action") and r.get("intention")]
    errors = [r for r in rows if not (r.get("action") and r.get("intention"))]
    action = [r["action"] for r in scored]
    intent = [r["intention"] for r in scored]
    hard = [r for r in scored if r["action"]["hard_success"]]

    def tier_rate(tier: str, turns: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        rates, skipped = [], 0
        for r in turns:
            bucket = r["action"]["by_tier"].get(tier)
            total = sum(bucket.values()) if bucket else 0
            if total:
                rates.append(bucket["satisfied"] / total)
            else:
                skipped += 1
        return metric(rates, skipped)

    changes = [i["change"] for i in intent if i["change"]]
    pairs = [p for i in intent for p in i["tier_pairs"]]
    feas = [a["feasibility"]["status"] for a in action]
    return {
        "turns": len(rows),
        "scored_turns": len(scored),
        "judge_errors": len(errors),
        "gold_plan_missing": feas.count("gold_plan_missing"),
        "not_feasible_turns": feas.count("not_feasible"),
        "action": {
            "hard_success": metric([float(a["hard_success"]) for a in action]),
            "strict_success": metric([float(a["strict_success"]) for a in action]),
            "hotel_gate_fail": metric([float(not a["hotel"]["valid"]) for a in action]),
            "min_nights_fail": metric([float(bool(a["hotel"]["minimum_nights"])) for a in action]),
            "capacity_fail": metric([float(bool(a["hotel"]["capacity"])) for a in action]),
            "undisclosed_must_violation": metric([float(bool(a["undisclosed_violated_musts"])) for a in action]),
            "must_violation_disclosure": {
                "value": _ratio(sum(len(set(a["violated_musts"]) & set(a["disclosed"])) for a in action),
                                sum(len(a["violated_musts"]) for a in action)),
                "numerator": sum(len(set(a["violated_musts"]) & set(a["disclosed"])) for a in action),
                "denominator": sum(len(a["violated_musts"]) for a in action),
                "excluded": 0,
            },
            "budget_violation_disclosure": {
                "value": _ratio(sum("budget" in a["violated_musts"] and "budget" in a["disclosed"] for a in action),
                                sum("budget" in a["violated_musts"] for a in action)),
                "numerator": sum("budget" in a["violated_musts"] and "budget" in a["disclosed"] for a in action),
                "denominator": sum("budget" in a["violated_musts"] for a in action),
                "excluded": 0,
            },
            "must_satisfaction": tier_rate("must_have", scored),
            "preferred_after_hard_success": tier_rate("preferred", hard),
            "optional_after_hard_success": tier_rate("optional", hard),
            "budget_satisfied": metric([float(a["budget"]["status"] == "satisfied") for a in action if a["budget"]],
                                       sum(1 for a in action if not a["budget"])),
            "budget_coverage_gap": metric([float(bool(a["budget"]["coverage_gaps"])) for a in action if a["budget"]]),
            "out_of_scope_constraints": sum(len(a["out_of_scope"]) for a in action),
            "by_shard": {
                shard: {
                    "hard_success": sum(r["action"]["hard_success"] for r in scored if r["shard"] == shard),
                    "turns": sum(1 for r in rows if r["shard"] == shard),
                }
                for shard in shards
            },
        },
        "intention": {
            "precision": metric([i["precision"] for i in intent]),
            "recall": metric([i["recall"] for i in intent]),
            "f1": metric([i["f1"] for i in intent]),
            "micro_precision": _ratio(sum(i["correct"] for i in intent), sum(i["atoms_pred"] for i in intent)),
            "micro_recall": _ratio(sum(i["correct"] for i in intent), sum(i["atoms_gold"] for i in intent)),
            "turn_exact": metric([float(i["turn_exact"]) for i in intent]),
            "change_precision": metric([c["precision"] for c in changes], len(intent) - len(changes)),
            "change_recall": metric([c["recall"] for c in changes], len(intent) - len(changes)),
            "change_f1": metric([c["f1"] for c in changes], len(intent) - len(changes)),
            "priority_accuracy": {"value": _ratio(sum(g == p for g, p in pairs), len(pairs)),
                                  "numerator": sum(g == p for g, p in pairs), "denominator": len(pairs), "excluded": 0},
            "priority_turn_exact": metric([float(i["priority_turn_exact"]) for i in intent]),
        },
    }
