#!/usr/bin/env python3
"""Draw the third-party evaluation pool and write the site's static data.

Reads annotated TravelPlanner and/or WebShop trajectory files, draws a seeded
random pool per domain, and writes evaluator-facing copies to public/data/:

  pool.json             pool version, settings, source hashes, item index
  items/<item_id>.json  one trajectory per file, loaded on demand by the site

Evaluators see the user turns, the gold labels they are asked to check, and the
evidence needed to check them. Model rollouts and predictions, annotator
rationales, and review notes (including disclosed violations) are dropped so
they cannot anchor the evaluator.

Example, from the repository root:

  python3 annotation/eval_site/build_pool.py \\
    --source 'annotation/data/travelplanner_v3_shards/shard_00[2378]_annotated.json' \\
    --size travelplanner=60 --seed 20260930
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path

SITE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SITE_DIR.parents[1]
DEFAULT_OUTPUT_DIR = SITE_DIR / "public" / "data"
DEFAULT_PRODUCT_CACHE = PROJECT_ROOT / "annotation" / "data" / "replay_item_cache.json"

DOMAIN_PREFIX = {"travelplanner": "tp", "webshop": "ws"}
SUBTYPE_DOMAIN = {"travel": "travelplanner", "shopping": "webshop"}
TRIP_FIELDS = ("org", "dest", "days", "visiting_city_number", "date", "people_number")
DESCRIPTION_LIMIT = 1500
TITLE_LIMIT = 140
ITEM_FILE_WARN_BYTES = 900_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def expand_sources(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) if any(ch in pattern for ch in "*?[") else [pattern]
        if not matches:
            raise SystemExit(f"No files match {pattern!r}")
        for match in matches:
            path = Path(match)
            if not path.is_file():
                raise SystemExit(f"Not a file: {match}")
            if path not in paths:
                paths.append(path)
    return paths


def load_trajectories(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("instances") or next((v for v in data.values() if isinstance(v, list)), [])
    return [t for t in data if isinstance(t, dict) and isinstance(t.get("turns"), list)]


def trajectory_domain(trajectory: dict) -> str | None:
    domain = (trajectory.get("world_state") or {}).get("domain")
    return domain if domain in DOMAIN_PREFIX else SUBTYPE_DOMAIN.get(trajectory.get("subtype"))


def make_item_id(domain: str, instance_id: str) -> str:
    # Independent of the source file, so ids survive data moving between files.
    raw = f"{DOMAIN_PREFIX[domain]}-{instance_id}"
    # Judgment documents are keyed "<code>__<item_id>", so item ids must not contain "__".
    return re.sub(r"_{2,}", "_", re.sub(r"[^A-Za-z0-9_-]+", "-", raw))


def choose_eval_turns(n_turns: int, item_id: str, eval_turns: int | None, include_seed: bool, seed: int) -> list[int]:
    """Turn positions to review; sampled once per item so every evaluator reviews the same turns."""
    candidates = list(range(0 if include_seed else 1, n_turns))
    if eval_turns is None or eval_turns >= len(candidates):
        return candidates
    return sorted(random.Random(f"{seed}:{item_id}").sample(candidates, eval_turns))


def normalize_priority(priority) -> dict:
    priority = priority if isinstance(priority, dict) else {}
    return {tier: list(priority.get(tier) or []) for tier in ("high", "medium", "low")}


def clean_delta(delta) -> dict:
    if not isinstance(delta, dict):
        return {}
    cleaned = {}
    for field, change in delta.items():
        if isinstance(change, dict):
            cleaned[field] = {key: change.get(key) for key in ("op", "old", "new") if key in change}
        else:
            cleaned[field] = {"op": "set", "new": change}
    return cleaned


def clean_text(value) -> str:
    # Catalog scrapes carry layout whitespace and invisible direction marks.
    return re.sub(r"\s+", " ", re.sub(r"[\u200e\u200f]", "", str(value))).strip()


def summarize_product(product) -> dict | None:
    if not isinstance(product, dict):
        return None
    description = clean_text(product.get("description") or "")
    if len(description) > DESCRIPTION_LIMIT:
        description = description[:DESCRIPTION_LIMIT].rstrip() + "…"
    information = product.get("product_information")
    return {
        "title": product.get("title"),
        "price": product.get("price"),
        "list_price": product.get("list_price"),
        "category": product.get("product_category") or product.get("category"),
        "brand": product.get("brand"),
        "rating": product.get("average_rating"),
        "reviews": product.get("total_reviews"),
        "options": product.get("options") or {},
        "attributes": product.get("attributes") or [],
        "bullet_points": product.get("bullet_points") or [],
        "information": {clean_text(k): clean_text(v) for k, v in information.items()}
        if isinstance(information, dict)
        else {},
        "description": description,
        "image_url": product.get("image_url"),
    }


def travel_action(action) -> dict | None:
    if not isinstance(action, dict):
        return None
    plan = (action.get("action_payload") or {}).get("plan") or {}
    return {
        "kind": "travel_plan",
        "itinerary": plan.get("itinerary") or [],
        "cost_ledger": action.get("cost_ledger") or [],
        "known_cost_subtotal": action.get("known_cost_subtotal"),
        "not_in_subtotal": action.get("unpriced_or_unverified") or [],
    }


def shop_action(action, products: dict) -> dict | None:
    if not isinstance(action, dict):
        return None
    payload = action.get("action_payload") or {}
    asin = payload.get("selected_asin")
    if not asin:
        return None
    return {
        "kind": "product",
        "asin": asin,
        "selected_options": payload.get("selected_options") or {},
        "product": summarize_product(products.get(asin)),
    }


def convert_turn(turn: dict, domain: str, evaluate: bool, products: dict) -> dict:
    gold = turn.get("gold_current_intention") or {}
    action = turn.get("gold_action")
    return {
        "turn_id": turn.get("turn_id"),
        "utterance": turn.get("user_utterance") or "",
        "evaluate": evaluate,
        "gold": {
            "state": {
                "constraints": gold.get("constraints") or {},
                "priority": normalize_priority(gold.get("priority")),
                "entities": gold.get("entities") or {},
            },
            "delta": clean_delta(turn.get("gold_delta")),
            "action": travel_action(action) if domain == "travelplanner" else shop_action(action, products),
        },
    }


def convert_item(candidate: dict, products: dict) -> dict:
    trajectory, domain = candidate["trajectory"], candidate["domain"]
    world = trajectory.get("world_state") or {}
    evaluated = set(candidate["eval_turns"])
    context, evidence = {}, {}
    if domain == "travelplanner":
        query = world.get("travelplanner_query_data") or {}
        context = {"trip": {key: query[key] for key in TRIP_FIELDS if key in query}}
        evidence = {"reference_information": world.get("reference_information") or {}}
    return {
        "item_id": candidate["item_id"],
        "domain": domain,
        "instance_id": candidate["instance_id"],
        "source": candidate["source"],
        "context": context,
        "evidence": evidence,
        "turns": [
            convert_turn(turn, domain, index in evaluated, products)
            for index, turn in enumerate(trajectory["turns"])
        ],
    }


def unconfirmed_turns(trajectory: dict, positions: list[int]) -> list[int]:
    return [i for i in positions if not (trajectory["turns"][i].get("gold_action") or {}).get("confirmed")]


def parse_sizes(values: list[str] | None) -> dict[str, int]:
    sizes = {}
    for value in values or []:
        domain, _, count = value.partition("=")
        if domain not in DOMAIN_PREFIX or not count.isdigit() or int(count) < 1:
            raise SystemExit(f"--size expects DOMAIN=N with DOMAIN in {sorted(DOMAIN_PREFIX)}; got {value!r}")
        sizes[domain] = int(count)
    return sizes


def parse_eval_turns(value: str) -> int | None:
    if value == "all":
        return None
    if value.isdigit() and int(value) >= 1:
        return int(value)
    raise argparse.ArgumentTypeError("expected 'all' or a positive whole number")


def collect_candidates(sources: list[Path], args) -> tuple[dict, list[dict], list[str]]:
    by_domain: dict[str, list[dict]] = {}
    source_records, notes = [], []
    seen: set[tuple[str, str]] = set()
    for path in sources:
        counts: dict[str, int] = {}
        for trajectory in load_trajectories(path):
            domain = trajectory_domain(trajectory)
            instance_id = str(trajectory.get("instance_id") or "")
            if domain is None or not instance_id:
                notes.append(f"{path.name}: skipped a trajectory without a known domain or instance_id")
                continue
            if (domain, instance_id) in seen:
                notes.append(f"{path.name}: skipped {instance_id}, already taken from an earlier source")
                continue
            seen.add((domain, instance_id))
            item_id = make_item_id(domain, instance_id)
            eval_turns = choose_eval_turns(
                len(trajectory["turns"]), item_id, args.eval_turns, args.include_seed_turn, args.seed
            )
            if not eval_turns:
                notes.append(f"{path.name}: skipped {instance_id}, no turns to review")
                continue
            if args.require_confirmed and unconfirmed_turns(trajectory, eval_turns):
                notes.append(f"{path.name}: skipped {instance_id}, unconfirmed gold actions")
                continue
            counts[domain] = counts.get(domain, 0) + 1
            by_domain.setdefault(domain, []).append(
                {
                    "item_id": item_id,
                    "domain": domain,
                    "instance_id": instance_id,
                    "source": display_path(path),
                    "trajectory": trajectory,
                    "eval_turns": eval_turns,
                }
            )
        source_records.append({"path": display_path(path), "sha256": sha256_file(path), "eligible": counts})
    return by_domain, source_records, notes


def draw(by_domain: dict, sizes: dict[str, int], seed: int) -> tuple[dict, list[str]]:
    drawn, notes = {}, []
    for domain in sorted(by_domain):
        # Sort before sampling so the draw depends only on the seed, not on file order.
        candidates = sorted(by_domain[domain], key=lambda c: c["item_id"])
        wanted = sizes.get(domain, len(candidates))
        if wanted > len(candidates):
            notes.append(f"{domain}: asked for {wanted} trajectories but only {len(candidates)} are eligible")
            wanted = len(candidates)
        picked = random.Random(f"{seed}:{domain}").sample(candidates, wanted)
        drawn[domain] = sorted(picked, key=lambda c: c["item_id"])
    for domain in sorted(set(sizes) - set(by_domain)):
        notes.append(f"{domain}: --size given but no eligible trajectories were found")
    return drawn, notes


def load_products(path: Path) -> dict:
    if not path.is_file():
        print(f"warning: product cache {display_path(path)} not found; WebShop products will show the ASIN only")
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("items", data) if isinstance(data, dict) else {}


def pool_version(drawn: dict, seed: int, created: datetime) -> str:
    fingerprint = json.dumps(
        {"seed": seed, "items": {d: [[c["item_id"], c["eval_turns"]] for c in items] for d, items in drawn.items()}},
        sort_keys=True,
    )
    return f"pool-{created:%Y%m%d}-{hashlib.sha256(fingerprint.encode()).hexdigest()[:8]}"


def build(args) -> tuple[dict, list[dict], list[str]]:
    by_domain, source_records, notes = collect_candidates(expand_sources(args.source), args)
    drawn, draw_notes = draw(by_domain, parse_sizes(args.size), args.seed)
    notes += draw_notes
    products = load_products(Path(args.product_cache)) if drawn.get("webshop") else {}
    items = [convert_item(c, products) for domain in sorted(drawn) for c in drawn[domain]]
    created = datetime.now(timezone.utc)
    pool = {
        "pool_version": pool_version(drawn, args.seed, created),
        "created_at": created.isoformat(timespec="seconds"),
        "demo": args.demo,
        "seed": args.seed,
        "eval_turns": "all" if args.eval_turns is None else args.eval_turns,
        "include_seed_turn": args.include_seed_turn,
        "require_confirmed": args.require_confirmed,
        "sources": source_records,
        "domains": {
            domain: {
                "eligible": len(by_domain.get(domain, [])),
                "items": [
                    {
                        "item_id": c["item_id"],
                        "instance_id": c["instance_id"],
                        "source": c["source"],
                        "n_turns": len(c["trajectory"]["turns"]),
                        "eval_turns": c["eval_turns"],
                        "title": textwrap.shorten(
                            c["trajectory"]["turns"][0].get("user_utterance") or "", TITLE_LIMIT, placeholder=" …"
                        ),
                    }
                    for c in candidates
                ],
            }
            for domain, candidates in drawn.items()
        },
    }
    return pool, items, notes


def write_pool(output_dir: Path, pool: dict, items: list[dict]) -> list[str]:
    items_dir = output_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    for stale in items_dir.glob("*.json"):
        stale.unlink()
    warnings = []
    for item in items:
        text = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        if len(text.encode("utf-8")) > ITEM_FILE_WARN_BYTES:
            warnings.append(f"{item['item_id']}: item file is {len(text) // 1000} KB")
        (items_dir / f"{item['item_id']}.json").write_text(text, encoding="utf-8")
    (output_dir / "pool.json").write_text(json.dumps(pool, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return warnings


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", action="append", required=True, help="Annotated JSON file or glob; repeatable.")
    parser.add_argument("--size", action="append", help="DOMAIN=N trajectories to draw (default: all eligible); repeatable.")
    parser.add_argument("--seed", type=int, default=20260930, help="Seed for the pool draw and turn sampling.")
    parser.add_argument(
        "--eval-turns",
        type=parse_eval_turns,
        default=None,
        help="'all' (default) or N: review N sampled turns per trajectory; every evaluator reviews the same turns.",
    )
    parser.add_argument("--include-seed-turn", action="store_true", help="Also review turn 0 (the original request).")
    parser.add_argument(
        "--require-confirmed", action="store_true", help="Skip trajectories with unconfirmed gold actions on reviewed turns."
    )
    parser.add_argument("--product-cache", default=str(DEFAULT_PRODUCT_CACHE), help="WebShop product cache JSON.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--demo", action="store_true", help="Mark the pool as demo data; the site shows a banner.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    pool, items, notes = build(args)
    if not items:
        for note in notes:
            print(f"note: {note}")
        raise SystemExit("No trajectories were drawn; nothing written.")
    warnings = write_pool(Path(args.output_dir), pool, items)
    print(f"{pool['pool_version']} -> {display_path(Path(args.output_dir))}")
    for domain, entry in pool["domains"].items():
        print(f"  {domain}: drew {len(entry['items'])} of {entry['eligible']} eligible trajectories")
    for note in notes + warnings:
        print(f"note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
