import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval import travelplanner_eval_v2 as V2
from eval.travelplanner_checks import hotel_validity, meal_coverage_gaps, trip_cost

REFERENCE = {
    "Accommodations in X": [
        {"NAME": "Big Loft", "price": 100, "minimum nights": 2, "maximum occupancy": 3, "room type": "Entire home/apt"},
        {"NAME": "Tiny Room", "price": 50, "minimum nights": 1, "maximum occupancy": 1, "room type": "Private room"},
    ],
    "Restaurants in X": [
        {"Name": "Cafe One", "Average Cost": 20, "Aggregate Rating": 4.2, "Cuisines": "Cafe"},
    ],
}


def test_group_split_across_listings_passes_capacity_and_is_priced_per_unit():
    itinerary = [
        {"accommodation": "Travelers 1-3: Big Loft; Traveler 4: Tiny Room"},
        {"accommodation": "Big Loft + Tiny Room"},
        {"accommodation": "-"},
    ]
    assert hotel_validity(itinerary, REFERENCE, 4)["valid"]
    assert trip_cost(itinerary, REFERENCE, 4)["total"] == 300


def test_minimum_nights_counts_consecutive_nights_per_listing():
    one_night = [{"accommodation": "Big Loft"}, {"accommodation": "-"}]
    result = hotel_validity(one_night, REFERENCE, 1)
    assert not result["valid"]
    assert result["minimum_nights"] == ["Big Loft: 1 consecutive night(s) booked, minimum 2"]


def test_meal_slot_priced_by_gold_but_self_catered_by_agent_is_a_gap():
    gold = [{"breakfast": "-", "lunch": "Cafe One"}]
    agent = [{"breakfast": "-", "lunch": "packed lunch"}]
    assert meal_coverage_gaps(agent, gold, REFERENCE) == ["day 1 lunch: gold 'cafe one', agent 'packed lunch'"]
    assert meal_coverage_gaps([{"lunch": "Cafe One"}], gold, REFERENCE) == []


def test_fewshot_leaves_out_the_judged_instance():
    cases = [
        {"stage": "action", "use_as_fewshot": True, "source": {"instance_id": "a"}, "lesson": "L1",
         "field": "f", "gold_value": 1, "tier": "must_have", "evidence": "e", "expected": {"action_status": "satisfied"}},
        {"stage": "action", "use_as_fewshot": True, "source": {"instance_id": "b"}, "lesson": "L2",
         "field": "f", "gold_value": 1, "tier": "must_have", "evidence": "e", "expected": {"action_status": "violated"}},
    ]
    block = V2.fewshot_block("action", "a", cases)
    assert "L2" in block and "L1" not in block


def _baseline(violated_by_gold):
    return {
        "has_gold_plan": True,
        "out_of_scope_fields": [],
        "gold_plan_code": {"budget": None, "hotel": {"valid": True}},
        "judge": {"gold_plan_judgments": [
            {"gold_field": f, "status": "violated" if f in violated_by_gold else "satisfied"} for f in ("room_type", "rating")
        ]},
    }


def _score(baseline, statuses, disclosed=()):
    gold = {"constraints": {"room_type": "Entire home/apt", "rating": 4},
            "priority": {"high": ["room_type", "rating"], "medium": [], "low": []}}
    judgment = {
        "final_plan": [],
        "constraint_judgments": [{"gold_field": f, "action_status": s} for f, s in statuses.items()],
        "unmet_constraint_disclosures": [{"gold_field": f, "quote": "q"} for f in disclosed],
    }
    return V2.score_action(gold=gold, baseline=baseline, judgment=judgment,
                           itinerary=[{"accommodation": "Big Loft"}, {"accommodation": "Big Loft"}],
                           reference=REFERENCE, gold_plan=None, people=1)


def test_not_feasible_turn_passes_only_with_disclosed_violation_of_gold_infeasible_must():
    baseline = _baseline({"rating"})
    assert _score(baseline, {"room_type": "satisfied", "rating": "violated"}, disclosed=["rating"])["hard_success"]
    assert not _score(baseline, {"room_type": "satisfied", "rating": "violated"})["hard_success"]
    # A Must the gold plan satisfies may not be traded away, disclosed or not.
    assert not _score(baseline, {"room_type": "violated", "rating": "satisfied"}, disclosed=["room_type"])["hard_success"]


def test_feasible_turn_requires_every_must():
    baseline = _baseline(set())
    assert _score(baseline, {"room_type": "satisfied", "rating": "satisfied"})["hard_success"]
    assert not _score(baseline, {"room_type": "satisfied", "rating": "violated"}, disclosed=["rating"])["hard_success"]


DIALOGUE = [
    {"turn": 0, "user": "Plan 3 days from Tampa to Cleveland, budget $1,800."},
    {"turn": 1, "user": "I'd rather pass through Gunnison on the way home, and keep lunch cheap."},
]


def test_quote_must_come_from_the_user_turns_allowing_ellipsis():
    assert V2.quote_in_dialogue("pass through Gunnison ... on the way home", DIALOGUE)
    assert not V2.quote_in_dialogue("Gunnison is only a road connection that day", DIALOGUE)
    assert not V2.quote_in_dialogue("", DIALOGUE)


def _baseline_raw(**overrides):
    raw = {
        "gold_atoms": [{"atom_id": "g1", "source_field": "budget", "value": 1800},
                       {"atom_id": "g2", "source_field": "lunch", "value": "cheap"}],
        "activity_requirements": [],
        "out_of_scope": [],
        "constraint_criteria": [{"gold_field": "lunch", "criteria": "Every lunch is inexpensive.", "quote": "keep lunch cheap"}],
        "gold_plan_judgments": [],
        "annotation_issues": [],
    }
    raw.update(overrides)
    return raw


def test_baseline_requires_criteria_and_verbatim_quotes():
    gold = {"constraints": {"budget": 1800, "lunch": "cheap"}}
    assert V2.validate_baseline(_baseline_raw(), gold, False, DIALOGUE)
    for bad in (
        _baseline_raw(constraint_criteria=[]),
        _baseline_raw(constraint_criteria=[{"gold_field": "lunch", "criteria": "x", "quote": "invented words"}]),
        _baseline_raw(activity_requirements=[{"kind": "exclude", "quote": "no sightseeing in Gunnison"}]),
    ):
        try:
            V2.validate_baseline(bad, gold, False, DIALOGUE)
        except ValueError:
            continue
        raise AssertionError("invalid baseline accepted")
