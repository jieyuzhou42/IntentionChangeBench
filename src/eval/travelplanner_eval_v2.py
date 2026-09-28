"""TravelPlanner evaluation v2: frozen baseline, per-turn judges, scoring in code.

Three judge calls, all governed by ``rules/travelplanner_v2.md``:

* baseline   once per (instance, turn), shared by every model: gold atoms,
             activity requirements, out-of-scope fields, the gold plan's
             constraint verdicts and annotation issues (human feasibility is
             loaded separately, never inferred by the judge);
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
from eval.action_scoring import score_action_constraints
from eval.baseline import constraint_tiers, validate_scoring_priorities
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

RULES_VERSION = "travelplanner-rules-v3.1"
INTENTION_SCORING_VERSION = "intention-turn-macro-priority-v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = Path(__file__).with_name("rules") / "travelplanner_v2.md"
CALIBRATION_PATH = REPO_ROOT / "annotation" / "data" / "travelplanner_judge_calibration_v1.json"
AUDIT_PATH = REPO_ROOT / "annotation" / "data" / "exports" / "travelplanner_v4" / "gold_audit_v1.json"

GOLD_TIER = {"high": "must_have", "medium": "preferred", "low": "optional"}
TIERS = ("must_have", "preferred", "optional")
ACTION_STATUSES = {"satisfied", "violated", "unknown"}
CHANGE_STATUSES = {"new", "changed", "unchanged"}
VALUE_CHANGE_OPS = {"add", "override", "relax"}
PRIORITY_CHANGE_OPS = VALUE_CHANGE_OPS | {"reprioritize", "scope_correction"}
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
    return json.loads(path.read_text(encoding="utf-8"))["cases"] if path.exists() else []


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
    return constraint_tiers(gold)


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
    validate_scoring_priorities(gold, out_of_scope)
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


def prediction_for_judge(items, indexed=False):
    """Preserve explicit scope and priority for semantic/change judgments."""
    keys = ("field", "value", "priority", "entity", "entity_id", "scope", "reference")
    return [{**({"index": n} if indexed else {}),
             **{k: item[k] for k in keys if k in item}}
            for n, item in enumerate(items)]


def build_intention_prompt(payload: Dict[str, Any], rules: Dict[str, str]) -> str:
    schema = {
        "pred_atoms": [{"source_index": 0, "field": "...", "value": "one requirement",
                        "gold_atom_id": "g1 or null", "value_match": True, "scope_match": True,
                        "change_vs_previous": "new | changed | unchanged",
                        "priority_change_vs_previous": "new | changed | unchanged"}]
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
            atom["gold_atom_id"] = gid
        if atom.get("change_vs_previous") not in CHANGE_STATUSES:
            raise ValueError(f"intention: bad change status {atom.get('change_vs_previous')}")
        if atom.get("priority_change_vs_previous") not in CHANGE_STATUSES:
            raise ValueError("intention: missing/invalid priority change status; rejudge with current rules")
        for key in ("value_match", "scope_match"):
            if type(atom.get(key)) is not bool:
                raise ValueError("intention: missing/invalid " + key + "; rejudge with current rules")
        if atom["gold_atom_id"] is None and atom["scope_match"]:
            raise ValueError("intention: unmatched atom cannot have scope_match=true")
    if sources != set(range(n_items)):
        raise ValueError(f"intention: atoms cover items {sorted(sources)}, expected {n_items}")
    if first_turn:
        for atom in atoms:
            atom["change_vs_previous"] = "new"
            atom["priority_change_vs_previous"] = "new"
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
    """Human feasibility (default true), with Gold violations as diagnostics."""
    code = baseline.get("gold_plan_code") or {}
    human = baseline.get("world_feasible")
    if human is None:
        human = True
    if type(human) is not bool:
        raise ValueError("Invalid human World Feasibility; rebuild the baseline")
    tiers = gold_tiers(gold)
    out = set(baseline["out_of_scope_fields"])
    violated = {
        j["gold_field"] for j in baseline["judge"].get("gold_plan_judgments") or []
        if j.get("status") == "violated" and tiers.get(j["gold_field"]) == "must_have" and j["gold_field"] not in out
    }
    if (code.get("budget") or {}).get("status") == "violated" and tiers.get("budget") == "must_have":
        violated.add("budget")
    return {"status": "feasible" if human is True else "not_feasible" if human is False else "unlabeled",
            "world_feasible": human, "gold_plan_missing": not baseline.get("has_gold_plan"),
            "violated_musts": sorted(violated),
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
    validate_scoring_priorities(gold, out)
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

    gold_status = None
    if baseline.get("has_gold_plan"):
        gold_status = {j["gold_field"]: j["status"] for j in baseline["judge"].get("gold_plan_judgments") or []}
        gold_budget = (baseline.get("gold_plan_code") or {}).get("budget")
        if gold_budget is not None:
            gold_status["budget"] = gold_budget["status"]
    scored = score_action_constraints(tiers=tiers, agent=status, gold=gold_status,
        world_feasible=feas["world_feasible"], out_of_scope=out, action_valid=hotel["valid"])
    hard_success = scored["gate_pass"]

    by_tier: Dict[str, Dict[str, int]] = {}
    for f in in_scope:
        tier = tiers.get(f, "entity")
        bucket = by_tier.setdefault(tier, {"satisfied": 0, "violated": 0, "unknown": 0})
        bucket[status.get(f, "unknown")] += 1
    return {
        **scored,
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


def changed_gold_fields(gold, gold_delta, include_priority=False):
    """Select current Gold fields; delta never supplies scoring values/tiers."""
    fields, tiers = set(gold_constraints(gold)), gold_tiers(gold)
    selected = set()

    def visit(delta, prefix=""):
        for name, change in (delta or {}).items():
            if not isinstance(change, dict):
                continue
            if name == "entities":
                for eid, entity_delta in change.items():
                    visit(entity_delta, "entities." + str(eid) + ".constraints.")
                continue
            if name == "constraints" and "op" not in change:
                visit(change, prefix)
                continue
            op = change.get("op")
            is_priority_map = name == "priority" or name.endswith(".priority")
            if include_priority and is_priority_map and op == "reprioritize":
                priority_prefix = prefix
                if name.endswith(".priority"):
                    priority_prefix = name[:-len("priority")] + "constraints."
                old = change.get("old") or {}
                new = change.get("new") or {}
                if not isinstance(old, dict) or not isinstance(new, dict):
                    raise ValueError("reprioritize priority map must contain old/new tier mappings")
                old_tiers = {}
                candidates = set()
                for mapping in (old, new):
                    for tier, names in mapping.items():
                        tier = GOLD_TIER.get(tier, tier)
                        for field in names:
                            field = priority_prefix + re.sub(r"^constraints\.", "", str(field))
                            candidates.add(field)
                            if mapping is old:
                                old_tiers[field] = tier
                selected.update(f for f in candidates if f in fields and old_tiers.get(f) != tiers.get(f))
            elif op in (PRIORITY_CHANGE_OPS if include_priority else VALUE_CHANGE_OPS):
                field = prefix + re.sub(r"^constraints\.", "", name)
                if field in fields:
                    selected.add(field)

    visit(gold_delta)
    return selected


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
    atoms = validate_intention(judgment, len(items), set(gold_atoms), first_turn)["pred_atoms"]
    if any(tiers.get(str(a["source_field"])) not in TIERS for a in gold_atoms.values()):
        raise ValueError("Every Intention Gold atom requires a valid priority")
    def content_correct(atom):
        return atom.get("gold_atom_id") is not None and atom["value_match"] and atom["scope_match"]
    correct = [a for a in atoms if content_correct(a)]
    precision = _ratio(len(correct), len(atoms)) or 0.0
    recall = _ratio(len(correct), len(gold_atoms)) or 0.0

    def tier_of(atom: Dict[str, Any]) -> Optional[str]:
        index = int(atom["source_index"])
        return items[index].get("priority") if index < len(items) else None

    def jointly_correct(atom):
        return content_correct(atom) and tier_of(atom) == tiers[str(gold_atoms[str(atom['gold_atom_id'])]['source_field'])]

    joint = sum(jointly_correct(a) for a in atoms)
    pp = _ratio(joint, len(atoms)) or 0.0
    pr = _ratio(joint, len(gold_atoms)) or 0.0

    def change_score(with_priority):
        changed_fields = changed_gold_fields(gold, gold_delta, with_priority)
        targets = {gid for gid, g in gold_atoms.items() if str(g["source_field"]) in changed_fields}
        if first_turn or not targets:
            return None
        predictions = []
        for atom in atoms:
            gid = atom.get("gold_atom_id")
            changed = atom["change_vs_previous"] != "unchanged"
            if with_priority:
                changed = changed or atom["priority_change_vs_previous"] != "unchanged"
            # Include current predictions for changed Gold even if the agent failed
            # to update them. Include spurious predicted changes as false positives.
            if gid in targets or (changed and not content_correct(atom)):
                predictions.append(atom)
            elif with_priority and changed and not jointly_correct(atom):
                predictions.append(atom)
        test = jointly_correct if with_priority else content_correct
        tp = sum(a.get("gold_atom_id") in targets and test(a) for a in predictions)
        p = _ratio(tp, len(predictions)) or 0.0
        r = tp / len(targets)
        return {"gold": len(targets), "predicted": len(predictions), "correct": tp,
                "precision": p, "recall": r, "f1": _f1(p, r)}
    return {
        "scoring_version": INTENTION_SCORING_VERSION,
        "atoms_gold": len(gold_atoms),
        "atoms_pred": len(atoms),
        "correct": len(correct),
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "turn_exact": len(correct) == len(atoms) == len(gold_atoms),
        "conditional_priority_accuracy": _ratio(joint, len(correct)),
        "priority_correct": joint,
        "priority_precision": pp,
        "priority_recall": pr,
        "priority_f1": _f1(pp, pr),
        "priority_turn_exact": joint == len(atoms) == len(gold_atoms),
        "change": change_score(False),
        "priority_change": change_score(True),
    }


def summarize(rows: Sequence[Dict[str, Any]], shards: Iterable[str]) -> Dict[str, Any]:
    """Turn-macro metrics with numerator / denominator / excluded counts."""
    def metric(values: List[float], excluded: int = 0) -> Dict[str, Any]:
        return {"value": _ratio(sum(values), len(values)), "numerator": sum(values),
                "denominator": len(values), "excluded": excluded}

    scored = [r for r in rows if r.get("action")]
    errors = [r for r in rows if not (r.get("action") and r.get("intention"))]
    action = [r["action"] for r in scored]
    intent = [r["intention"] for r in rows if r.get("intention")]
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

    if any(i.get("scoring_version") != INTENTION_SCORING_VERSION for i in intent):
        raise ValueError("Stale Intention scores; rejudge/rescore with current scope and priority rules")
    changes = [i["change"] for i in intent if i["change"]]
    priority_changes = [i["priority_change"] for i in intent if i["priority_change"] is not None]
    conditional = [i["conditional_priority_accuracy"] for i in intent if i["conditional_priority_accuracy"] is not None]
    feas = [a["feasibility"]["status"] for a in action]
    return {
        "turns": len(rows),
        "scored_turns": len(scored),
        "judge_errors": len(errors),
        "gold_plan_missing": sum(a["gold_assumed_perfect"] for a in action),
        "action_scored_turns": len(action),
        "intention_scored_turns": len(intent),
        "not_feasible_turns": feas.count("not_feasible"),
        "action": {
            "action_score": metric([a["action_score"] for a in action], len(rows) - len(action)),
            "action_success": metric([float(a["action_success"]) for a in action], len(rows) - len(action)),
            "must_gate": metric([float(a["must_gate"]) for a in action], len(rows) - len(action)),
            "hard_success": metric([float(a["hard_success"]) for a in action]),
            "strict_success": metric([float(a["strict_success"]) for a in action]),
            "hotel_gate_fail": metric([float(not a["hotel"]["valid"]) for a in action]),
            "min_nights_fail": metric([float(bool(a["hotel"]["minimum_nights"])) for a in action]),
            "capacity_fail": metric([float(bool(a["hotel"]["capacity"])) for a in action]),
            "undisclosed_must_violation": metric([float(bool(a["undisclosed_violated_musts"])) for a in action]),
            "must_violation_disclosure": metric(
                [len(set(a["violated_musts"]) & set(a["disclosed"])) / len(a["violated_musts"])
                 for a in action if a["violated_musts"]],
                len(rows) - sum(bool(a["violated_musts"]) for a in action)),
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
            "scoring_version": INTENTION_SCORING_VERSION,
            **{k: metric([i[k] for i in intent], len(rows) - len(intent))
               for k in ("precision", "recall", "f1", "priority_precision", "priority_recall", "priority_f1")},
            "turn_exact": metric([float(i["turn_exact"]) for i in intent], len(rows) - len(intent)),
            **{"change_" + k: metric([c[k] for c in changes], len(rows) - len(changes))
               for k in ("precision", "recall", "f1")},
            "conditional_priority_accuracy": metric(conditional, len(rows) - len(conditional)),
            **{"priority_change_" + k: metric([c[k] for c in priority_changes], len(rows) - len(priority_changes))
               for k in ("precision", "recall", "f1")},
            "priority_turn_exact": metric([float(i["priority_turn_exact"]) for i in intent], len(rows) - len(intent)),
        },
    }
