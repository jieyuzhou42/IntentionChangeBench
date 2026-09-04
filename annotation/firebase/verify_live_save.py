from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path


SITE_URL = "https://intentflow-45722.web.app"
FIRESTORE_DOCUMENTS = (
    "https://firestore.googleapis.com/v1/projects/intentflow-45722/"
    "databases/(default)/documents/webshop_annotations"
)


def request_json(
    url: str,
    *,
    method: str = "GET",
    data: dict | None = None,
    token: str | None = None,
) -> dict:
    headers = {}
    body = None
    if data is not None:
        body = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(
        urllib.request.Request(url, data=body, headers=headers, method=method),
        timeout=60,
    ) as response:
        payload = response.read()
        return json.loads(payload) if payload else {}


def firestore_string(value: str) -> dict:
    return {"stringValue": value}


def clean_turns(turns: list[dict]) -> list[dict]:
    cleaned = json.loads(json.dumps(turns, ensure_ascii=False))
    for turn in cleaned:
        turn.pop("rationales", None)
        feedback = turn.get("env_feedback") or {}
        for item in feedback.get("candidate_items") or []:
            if isinstance(item, dict):
                item.pop("image_url", None)
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify an anonymous live Firestore save.")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--instance-id", required=True)
    args = parser.parse_args()

    admin_token = os.environ["INTENTFLOW_ADMIN_TOKEN"]
    public_dir = Path(__file__).resolve().parent / "public"
    state = json.loads(
        (public_dir / "data" / f"shard_{args.shard:03d}.json").read_text(encoding="utf-8")
    )
    instance = next(item for item in state["instances"] if item["instance_id"] == args.instance_id)
    turns_json = json.dumps(
        clean_turns(instance["turns"]), ensure_ascii=False, separators=(",", ":")
    )

    config = request_json(f"{SITE_URL}/__/firebase/init.json")
    signup = request_json(
        f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={config['apiKey']}",
        method="POST",
        data={"returnSecureToken": True},
    )
    test_id = f"codex_{args.instance_id}_smoke"
    document_url = f"{FIRESTORE_DOCUMENTS}/{test_id}?key={config['apiKey']}"
    document = {
        "fields": {
            "instance_id": firestore_string(test_id),
            "shard_index": {"integerValue": str(args.shard)},
            "turns_json": firestore_string(turns_json),
            "deleted": {"booleanValue": False},
            "updated_at": {"timestampValue": "2026-09-04T00:00:00Z"},
        }
    }
    try:
        written = request_json(
            document_url, method="PATCH", data=document, token=signup["idToken"]
        )
        read_back = request_json(document_url, token=signup["idToken"])
        saved_json = read_back["fields"]["turns_json"]["stringValue"]
        print(
            json.dumps(
                {
                    "written": bool(written.get("name")),
                    "read_back": bool(read_back.get("name")),
                    "json_round_trip": saved_json == turns_json,
                    "turns_json_bytes": len(turns_json.encode("utf-8")),
                }
            )
        )
    finally:
        request_json(document_url, method="DELETE", token=admin_token)
        print("smoke_cleanup=true")


if __name__ == "__main__":
    main()
