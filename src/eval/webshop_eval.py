"""WebShop frozen baseline and action judgments, sharing the Action scorer."""
import json
from pathlib import Path

from eval import travelplanner_eval_v2 as intent
from eval.action_scoring import score_action_constraints
from eval.baseline import (BASELINE_VERSION, constraint_tiers, fingerprint, human_world_feasibility,
                           normalize_gold, validate_scoring_priorities)
from eval.webshop_checks import selection, validate_selection

RULES_VERSION = "webshop-rules-v1.0"
RULES_PATH = Path(__file__).with_name("rules") / "webshop.md"


def make_input(instance, turn_index, catalog):
    turn = instance["turns"][turn_index]
    gold = normalize_gold(turn.get("gold_current_intention") or {})
    if not gold.get("constraints"):
        raise ValueError("Missing Gold intention constraints")
    validate_scoring_priorities(gold)
    selected = selection(turn, gold=True)
    has_gold = bool(selected["asin"])
    human = human_world_feasibility(turn)
    checked = validate_selection(selected, catalog) if has_gold else None
    return {"domain": "webshop", "instance_id": instance["instance_id"], "turn_id": turn.get("turn_id", turn_index),
            "gold": gold, "gold_delta": turn.get("gold_delta") or {},
            "world_feasible": human, "has_gold_action": has_gold,
            "gold_selection": selected if has_gold else None, "gold_product": checked["product"] if checked else None,
            "gold_selection_valid": checked["valid"] if checked else None,
            "catalog_sha256": fingerprint(catalog), "dialogue": intent.dialogue_so_far(instance["turns"], turn_index)}


def baseline_prompt(payload):
    # Delta annotations can be stale or inconsistent with the current snapshot.
    # Keep them in the frozen input for change scoring, not baseline construction.
    judge_input = {k: v for k, v in payload.items() if k != "gold_delta"}
    schema = {"gold_atoms": [{"atom_id": "g1", "source_field": "exact field", "value": "one requirement"}],
              "constraint_criteria": [{"gold_field": "exact field", "criteria": "all required evidence", "quote": "verbatim user quote or empty"}],
              "gold_plan_judgments": [{"gold_field": "exact field", "status": "satisfied | violated | unknown", "evidence": "product field and supporting text"}],
              "annotation_issues": []}
    return (RULES_PATH.read_text(encoding="utf-8") + "\nPrepare a shared baseline, independent of tested models. "
            "Cover every active constraint exactly once in criteria; split gold_atoms only for intention matching. "
            "The sole authority for active constraint fields, values and priorities is INPUT.gold, "
            "normalized from gold_current_intention. Use exactly the fields in INPUT.gold.constraints. "
            "Dialogue, delta annotations, product facts and agent claims must never add, remove, "
            "replace or relax these constraints. Dialogue only clarifies their meaning. "
            "If another source disagrees, report an annotation issue and retain the current Gold snapshot. "
            "Judge every constraint against Gold selection if present, otherwise return no Gold judgments. "
            "Never judge World Feasibility or alter human tiers. Report annotation conflicts without repairing them. "
            "Return JSON matching: " + json.dumps(schema) + "\nINPUT:\n" + json.dumps(judge_input, ensure_ascii=False))


def validate_baseline(raw, payload):
    fields = set(payload["gold"]["constraints"])
    criteria = raw.get("constraint_criteria") or []
    if len(criteria) != len(fields) or {c["gold_field"] for c in criteria} != fields:
        raise ValueError("baseline criteria must cover every field exactly once")
    for c in criteria:
        if not c.get("criteria"):
            raise ValueError("Empty criterion")
        if c.get("quote") and not intent.quote_in_dialogue(c["quote"], payload["dialogue"]):
            raise ValueError("Criterion quote absent from dialogue")
    atoms = raw.get("gold_atoms") or []
    if {a["source_field"] for a in atoms} != fields or len({a["atom_id"] for a in atoms}) != len(atoms):
        raise ValueError("Invalid Gold atoms")
    judgments = raw.get("gold_plan_judgments") or []
    validate_statuses(judgments, fields if payload["has_gold_action"] else set(), "status")
    return raw


def validate_statuses(rows, fields, key):
    if len(rows) != len(fields) or {r["gold_field"] for r in rows} != fields:
        raise ValueError("Judgments must cover each constraint exactly once")
    for r in rows:
        if r.get(key) not in {"satisfied", "violated", "unknown"} or not r.get("evidence"):
            raise ValueError("Invalid status or missing evidence")


def action_prompt(baseline, selected, catalog):
    checked = validate_selection(selected, catalog)
    payload = {"criteria": baseline["judge"]["constraint_criteria"],
               "selected_action": selected, "product": checked["product"], "selection_valid": checked["valid"]}
    return (RULES_PATH.read_text(encoding="utf-8") + "\nAudit only this selected action against frozen criteria. "
            "Do not see/infer the agent intention or use Gold performance to change verdicts. "
            "Return JSON: {\"constraint_judgments\":[{\"gold_field\":\"exact field\","
            "\"action_status\":\"satisfied | violated | unknown\",\"evidence\":\"specific product evidence or missing fact\"}],"
            "\"unmet_constraint_disclosures\":[{\"gold_field\":\"field\",\"quote\":\"exact rationale quote\"}]}\n"
            + json.dumps(payload, ensure_ascii=False))


def validate_action(raw, fields):
    validate_statuses(raw.get("constraint_judgments") or [], fields, "action_status")
    return raw


def score_action(baseline, judgment, selected, catalog):
    payload = baseline["input"]
    tiers = constraint_tiers(payload["gold"])
    validate_action(judgment, set(tiers))
    agent = {r["gold_field"]: r["action_status"] for r in judgment["constraint_judgments"]}
    gold = ({r["gold_field"]: r["status"] for r in baseline["judge"]["gold_plan_judgments"]}
            if payload["has_gold_action"] else None)
    checked = validate_selection(selected, catalog)
    scored = score_action_constraints(tiers=tiers, agent=agent, gold=gold,
        world_feasible=payload["world_feasible"], action_valid=checked["valid"])
    return {**scored, "status": agent, "selection": selected, "selection_check": checked["reason"],
            "constraint_judgments": judgment["constraint_judgments"],
            "disclosures": judgment.get("unmet_constraint_disclosures") or []}
