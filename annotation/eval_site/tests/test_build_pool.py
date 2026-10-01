import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_pool  # noqa: E402

INTERNAL_KEYS = [
    "agent_action",
    "rollout_trace",
    "agent_intention_prediction",
    "annotation_review",
    "known_violations",
    "world_feasibility",
    "rationale",
    "gold_search_query",
]


def travel_trajectory(instance_id, n_turns=4, confirmed=False):
    turns = []
    for i in range(n_turns):
        turns.append(
            {
                "turn_id": i,
                "user_utterance": f"message {i}",
                "agent_action": {"plan": "model output"},
                "rollout_trace": [{"step": 1}],
                "agent_intention_prediction": {"budget": 1},
                "annotation_review": {"corrections": []},
                "gold_delta": {"budget": {"op": "override", "old": 100, "new": 100 + i, "rationale": "note"}} if i else {},
                "gold_current_intention": {
                    "constraints": {"budget": 100 + i},
                    "priority": {"high": ["budget"]},
                    "entities": {},
                    "gold_search_query": "q",
                },
                "gold_action": {
                    "action_type": "Planner",
                    "confirmed": confirmed,
                    "action_payload": {"plan": {"itinerary": [{"day": "2022-03-01"}]}},
                    "cost_ledger": [{"kind": "accommodation", "total": 10}],
                    "known_cost_subtotal": 10,
                    "unpriced_or_unverified": ["food"],
                    "known_violations": ["budget"],
                    "world_feasibility": {"feasible": True},
                    "rationale": "why",
                },
            }
        )
    return {
        "instance_id": instance_id,
        "subtype": "travel",
        "world_state": {
            "domain": "travelplanner",
            "travelplanner_query_data": {"org": "A", "dest": "B", "days": 3, "people_number": 2, "level": "hard"},
            "reference_information": {"Accommodations in B": [{"NAME": "H", "price": 10}]},
        },
        "turns": turns,
    }


def shop_trajectory(instance_id):
    state = {"constraints": {"category": "pillow"}, "priority": {"high": ["category"]}}
    return {
        "instance_id": instance_id,
        "subtype": "shopping",
        "world_state": {"domain": "webshop"},
        "turns": [
            {"turn_id": 0, "user_utterance": "a pillow", "gold_current_intention": state, "gold_action": None},
            {
                "turn_id": 1,
                "user_utterance": "any color is fine",
                "gold_delta": {"color": {"op": "remove", "old": "black", "new": None}},
                "gold_current_intention": state,
                "gold_action": {"action_type": "Buy", "action_payload": {"selected_asin": "B1", "selected_options": {"Size": "M"}}},
            },
        ],
    }


class BuildPoolTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.write("tp_a_annotated.json", [travel_trajectory(f"travel_{i}") for i in range(6)] + [travel_trajectory("odd__id/1")])
        self.write("tp_b_annotated.json", [travel_trajectory("travel_0"), travel_trajectory("travel_9", confirmed=True)])
        self.write("ws_human_annotated.json", [shop_trajectory("shop_0")])
        product = {"title": "Pillow", "price": 9.5, "description": "x" * 2000, "product_information": {"  Size \n\u200e": "\u200e M "}}
        self.write("products.json", {"items": {"B1": product}})

    def write(self, name, data):
        (self.dir / name).write_text(json.dumps(data), encoding="utf-8")

    def build(self, *extra, sources=("tp_*_annotated.json", "ws_human_annotated.json")):
        out = self.dir / "out"
        argv = [arg for source in sources for arg in ("--source", str(self.dir / source))]
        argv += ["--product-cache", str(self.dir / "products.json"), "--output-dir", str(out), *extra]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(build_pool.main(argv), 0)
        pool = json.loads((out / "pool.json").read_text(encoding="utf-8"))
        items = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in (out / "items").glob("*.json")}
        return pool, items

    def drawn(self, pool, domain="travelplanner"):
        return [item["item_id"] for item in pool["domains"][domain]["items"]]

    def test_draw_depends_only_on_the_seed(self):
        first, _ = self.build("--size", "travelplanner=3", "--seed", "5")
        again, _ = self.build("--size", "travelplanner=3", "--seed", "5")
        reordered, _ = self.build(
            "--size", "travelplanner=3", "--seed", "5", sources=("ws_human_annotated.json", "tp_b_annotated.json", "tp_a_annotated.json")
        )
        other, _ = self.build("--size", "travelplanner=3", "--seed", "6")
        self.assertEqual(len(self.drawn(first)), 3)
        self.assertEqual(self.drawn(first), self.drawn(again))
        self.assertEqual(first["pool_version"], again["pool_version"])
        self.assertNotEqual(self.drawn(first), self.drawn(other))
        self.assertEqual(self.drawn(first), self.drawn(reordered))

    def test_duplicate_instances_are_kept_once(self):
        pool, items = self.build()
        instances = [item["instance_id"] for item in pool["domains"]["travelplanner"]["items"]]
        self.assertEqual(len(instances), len(set(instances)))
        self.assertEqual(pool["domains"]["travelplanner"]["eligible"], 8)
        self.assertTrue(items["tp-travel_0"]["source"].endswith("tp_a_annotated.json"))

    def test_item_files_drop_model_outputs_and_review_notes(self):
        _, items = self.build()
        item = items["tp-travel_1"]
        text = json.dumps(item)
        for key in INTERNAL_KEYS:
            self.assertNotIn(f'"{key}"', text, key)
        action = item["turns"][1]["gold"]["action"]
        self.assertEqual(action["cost_ledger"], [{"kind": "accommodation", "total": 10}])
        self.assertEqual(action["not_in_subtotal"], ["food"])
        self.assertEqual(item["turns"][1]["gold"]["delta"], {"budget": {"op": "override", "old": 100, "new": 101}})
        self.assertNotIn("level", item["context"]["trip"])
        self.assertEqual(item["evidence"]["reference_information"], {"Accommodations in B": [{"NAME": "H", "price": 10}]})

    def test_seed_turn_is_context_unless_requested(self):
        pool, items = self.build()
        entry = pool["domains"]["travelplanner"]["items"][0]
        self.assertEqual(entry["eval_turns"], [1, 2, 3])
        self.assertEqual([t["evaluate"] for t in items[entry["item_id"]]["turns"]], [False, True, True, True])
        with_seed, _ = self.build("--include-seed-turn")
        self.assertEqual(with_seed["domains"]["travelplanner"]["items"][0]["eval_turns"], [0, 1, 2, 3])

    def test_sampled_turns_are_shared_and_reproducible(self):
        first, _ = self.build("--eval-turns", "2")
        again, _ = self.build("--eval-turns", "2")
        for entry, repeat in zip(first["domains"]["travelplanner"]["items"], again["domains"]["travelplanner"]["items"]):
            self.assertEqual(len(entry["eval_turns"]), 2)
            self.assertTrue(set(entry["eval_turns"]) <= {1, 2, 3})
            self.assertEqual(entry["eval_turns"], repeat["eval_turns"])

    def test_require_confirmed_skips_unconfirmed_gold_actions(self):
        pool, _ = self.build("--require-confirmed")
        self.assertEqual(self.drawn(pool), ["tp-travel_9"])
        self.assertNotIn("webshop", pool["domains"])

    def test_item_ids_are_safe_judgment_key_parts(self):
        pool, _ = self.build()
        ids = self.drawn(pool) + self.drawn(pool, "webshop")
        self.assertIn("tp-odd_id-1", ids)
        for item_id in ids:
            self.assertRegex(item_id, r"^[A-Za-z0-9_-]+$")
            self.assertNotIn("__", item_id)

    def test_webshop_products_come_from_the_cache(self):
        _, items = self.build()
        action = items["ws-shop_0"]["turns"][1]["gold"]["action"]
        self.assertEqual(action["asin"], "B1")
        self.assertEqual(action["selected_options"], {"Size": "M"})
        self.assertEqual(action["product"]["title"], "Pillow")
        self.assertEqual(action["product"]["information"], {"Size": "M"})
        self.assertEqual(len(action["product"]["description"]), build_pool.DESCRIPTION_LIMIT + 1)
        self.assertIsNone(items["ws-shop_0"]["turns"][0]["gold"]["action"])

    def test_pool_records_source_hashes(self):
        pool, _ = self.build()
        self.assertEqual(len(pool["sources"]), 3)
        for source in pool["sources"]:
            self.assertRegex(source["sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
