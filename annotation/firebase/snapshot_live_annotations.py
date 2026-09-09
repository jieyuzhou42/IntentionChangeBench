from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from annotation.firebase.verify_live_save import FIRESTORE_DOCUMENTS, SITE_URL, request_json


FIREBASE_DIR = Path(__file__).resolve().parent
PRESERVED_IDS_PATH = FIREBASE_DIR / "preserved_instance_ids.json"
BACKUP_DIR = FIREBASE_DIR / "backups"


def main() -> None:
    config = request_json(f"{SITE_URL}/__/firebase/init.json")
    signup = request_json(
        f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={config['apiKey']}",
        method="POST",
        data={"returnSecureToken": True},
    )
    result = request_json(
        f"{FIRESTORE_DOCUMENTS}?pageSize=300&key={config['apiKey']}",
        token=signup["idToken"],
    )
    documents = result.get("documents") or []
    now = datetime.now(timezone.utc)
    snapshot = {
        "project_id": "intentflow-45722",
        "retrieved_at": now.isoformat(),
        "documents": documents,
    }
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = BACKUP_DIR / f"webshop_annotations_{now:%Y%m%dT%H%M%SZ}.json"
    backup_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    preserved = []
    for document in documents:
        fields = document.get("fields") or {}
        instance_id = ((fields.get("instance_id") or {}).get("stringValue") or "").strip()
        shard = int((fields.get("shard_index") or {}).get("integerValue") or 0)
        deleted = bool((fields.get("deleted") or {}).get("booleanValue"))
        if instance_id and 6 <= shard <= 20 and not deleted and not instance_id.startswith("codex_"):
            preserved.append({"instance_id": instance_id, "shard": shard})
    preserved.sort(key=lambda item: (item["shard"], item["instance_id"]))
    PRESERVED_IDS_PATH.write_text(
        json.dumps(
            {"retrieved_at": now.isoformat(), "instances": preserved},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Backed up {len(documents)} documents to {backup_path}")
    print(f"Preserving {len(preserved)} annotated instances in {PRESERVED_IDS_PATH}")


if __name__ == "__main__":
    main()
